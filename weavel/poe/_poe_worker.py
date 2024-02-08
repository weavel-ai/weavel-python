from __future__ import annotations

import os
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from threading import Thread
from concurrent.futures import Future, ThreadPoolExecutor

from fastapi_poe import QueryRequest, PartialResponse

from weavel._constants import BACKEND_SERVER_URL
from weavel._api_client import APIClient
from weavel.utils import logger
from weavel.poe._poe_buffer_storage import PoeBufferStorage, PoeRequest, PoeSaveTraceDataBody

class PoeWorker:
    def __init__(
        self,
        api_key: str,
    ) -> None:
        self.api_key = api_key
        self.endpoint = BACKEND_SERVER_URL
        
        self.max_retry = 3
        self.flush_interval = 10
        self.flush_batch_size = 100
        
        self.api_client = APIClient()
        
        self.api_pool = ThreadPoolExecutor(max_workers=1)
        self.buffer_storage = PoeBufferStorage(max_buffer_size=1000)
        self._running = True
        self._thread = Thread(target=self.consume_buffer, daemon=True)
        self._thread.start()
    
    def log(
        self,
        user_request: QueryRequest,
        bot_responses: List[PartialResponse],
        response_timestamp: Optional[datetime] = datetime.now().isoformat(),
    ):
        pass
        user_log = PoeRequest(
            body=PoeSaveTraceDataBody(
                conversation_id=user_request.conversation_id,
                user_id=user_request.user_id,
                message_id=user_request.message_id,
                role="user",
                data_type="user_message",
                data_content=user_request.query[-1].content,
                timestamp=datetime.fromtimestamp(user_request.query[-1].timestamp / 1e6, timezone.utc).isoformat(),
            ),
        )
        bot_response_list: List[str] = [response.text for response in bot_responses]
        bot_response: str = "".join(bot_response_list)
        bot_log = PoeRequest(
            body=PoeSaveTraceDataBody(
                conversation_id=user_request.conversation_id,
                user_id=user_request.user_id,
                role="assistant",
                data_type="assistant_message",
                data_content=bot_response,
                timestamp=response_timestamp,
            ),
        )
        self.buffer_storage.push(user_log)
        self.buffer_storage.push(bot_log)
            
    def send_requests(self, requests: List[PoeRequest]):
        logger.info(requests)
        for attempt in range(self.max_retry):
            try:
                response = self.api_client.execute(
                    self.api_key,
                    self.endpoint,
                    "/poe/trace/batch",
                    method="POST",
                    json={"requests": [request.model_dump() for request in requests]},
                    timeout=10,
                )
                if response.status_code == 200:
                    return
            except Exception as e:
                print(e)
                time.sleep(2**attempt)
                continue
        
    def consume_buffer(self) -> None:
        while self._running:
            try:
                with self.buffer_storage.buffer_lock:
                    if self.buffer_storage.buffer_size == 0:
                        self.buffer_storage.not_empty_cv.wait(self.flush_interval)
                        continue
                    print("AWAKE!")
                    out = self.buffer_storage.pull(self.flush_batch_size)
                    future = self.api_pool.submit(self.send_requests, out)
                    future.add_done_callback(self._handle_api_task_completion)
                    self.buffer_storage.not_empty_cv.wait(self.flush_interval)
            except Exception as e:
                print(e)

    def flush(self) -> None:
        try:
            with self.buffer_storage.buffer_lock:
                if self.buffer_storage.buffer_size == 0:
                    return
                out = self.buffer_storage.pull_all()
                # for request chunks size of self.flush_batch_size
                for i in range(0, len(out), self.flush_batch_size):
                    request = out[i:i+self.flush_batch_size]
                    self.send_requests(request)
        except Exception as e:
            print(e)
            
    def stop(self) -> None:
        self._running = False
        with self.buffer_storage.buffer_lock:
            self.buffer_storage.not_empty_cv.notify()
        self._thread.join()
        self.flush()
                                     
    def _handle_api_task_completion(self, future: Future) -> None:
        try:
            future.result()
        except Exception as e:
            print(e)()