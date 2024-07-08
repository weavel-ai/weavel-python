from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from threading import Thread
from concurrent.futures import Future, ThreadPoolExecutor

from weavel._request import (
    CaptureSessionRequest,
    IdentifyUserRequest,
    CaptureMessageRequest,
    CaptureTrackEventRequest,
    CaptureTraceRequest,
    CaptureSpanRequest,
    CaptureLogRequest,
    CaptureGenerationRequest,
    UpdateTraceRequest,
    UpdateSpanRequest,
    UpdateGenerationRequest,
    UnionRequest
)
from weavel._body import (
    CaptureSessionBody,
    CaptureMessageBody,
    CaptureTrackEventBody,
    CaptureTraceBody,
    CaptureSpanBody,
    CaptureLogBody,
    CaptureGenerationBody,
    UpdateTraceBody,
    UpdateSpanBody,
    UpdateGenerationBody,
    IdentifyUserBody,
)
    
from weavel._constants import BACKEND_SERVER_URL
from weavel._buffer_storage import BufferStorage
from weavel._api_client import APIClient
from weavel.utils import logger


class Worker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Worker, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        api_key: str,
    ) -> None:
        if not hasattr(self, "is_initialized"):
            self.api_key = api_key
            self.endpoint = BACKEND_SERVER_URL

            self.max_retry = 3
            self.flush_interval = 60
            self.flush_batch_size = 20

            self.api_client = APIClient()

            self.api_pool = ThreadPoolExecutor(max_workers=1)
            self.buffer_storage = BufferStorage(max_buffer_size=1000)
            self._running = True
            self._thread = Thread(target=self.consume_buffer, daemon=True)
            self._thread.start()
            self.is_initialized = True

    def open_session(
        self,
        session_id: str,
        created_at: datetime,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start the new session for user_id.

        Returns:
            The session id.
        """
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = created_at

        # add task to buffer
        request = CaptureSessionRequest(
            body=CaptureSessionBody(
                user_id=user_id,
                session_id=session_id,
                created_at=created_at.isoformat(),
                metadata=metadata,
            )
        )

        self.buffer_storage.push(request)

        return
    
    def identify_user(self, user_id: str, properties: Dict) -> None:
        """
        Identifies a user with the given properties.

        Args:
            user_id (str): The ID of the user.
            properties (Dict): The properties associated with the user.

        Returns:
            None
        """
        request = IdentifyUserRequest(
            body=IdentifyUserBody(
                user_id=user_id,
                properties=properties,
                created_at=datetime.now(timezone.utc).isoformat()
            )
        )
        self.buffer_storage.push(request)

        return
    
    def capture_message(
        self,
        session_id: str,
        record_id: str,
        created_at: datetime,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ref_record_id: Optional[str] = None,
    ):
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = created_at
            
        request = CaptureMessageRequest(
            body=CaptureMessageBody(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at.isoformat(),
                role=role,
                content=content,
                metadata=metadata,
                ref_record_id=ref_record_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def capture_track_event(
        self,
        session_id: str,
        record_id: str,
        created_at: datetime,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ref_record_id: Optional[str] = None,
    ):
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = created_at
            
        request = CaptureTrackEventRequest(
            body=CaptureTrackEventBody(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at.isoformat(),
                name=name,
                properties=properties,
                metadata=metadata,
                ref_record_id=ref_record_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def capture_trace( 
        self,  
        session_id: str,
        record_id: str,
        created_at: datetime,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        ref_record_id: Optional[str] = None,
    ):
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = created_at
            
        request = CaptureTraceRequest(
            body=CaptureTraceBody(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at.isoformat(),
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                ref_record_id=ref_record_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def update_trace(
        self,
        record_id: str,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None, 
        metadata: Optional[Dict[str, str]] = None,
        ref_record_id: Optional[str] = None,
    ):
        request = UpdateTraceRequest(
            body=UpdateTraceBody(
                record_id=record_id,
                ended_at=ended_at.isoformat() if ended_at else None,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                ref_record_id=ref_record_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def capture_span(
        self,
        record_id: str,
        observation_id: str,
        created_at: datetime,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        request = CaptureSpanRequest(
            body=CaptureSpanBody(
                record_id=record_id,
                observation_id=observation_id,
                created_at=created_at.isoformat(),
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def update_span(
        self,
        observation_id: str,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        request = UpdateSpanRequest(
            body=UpdateSpanBody(
                observation_id=observation_id,
                ended_at=ended_at.isoformat() if ended_at else None,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def capture_log(
        self,
        record_id: str,
        observation_id: str,
        created_at: datetime,
        name: str,
        value: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        request = CaptureLogRequest(
            body=CaptureLogBody(
                record_id=record_id,
                observation_id=observation_id,
                created_at=created_at.isoformat(),
                name=name,
                value=value,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def capture_generation(
        self,
        record_id: str,
        observation_id: str,
        created_at: datetime,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        request = CaptureGenerationRequest(
            body=CaptureGenerationBody(
                record_id=record_id,
                observation_id=observation_id,
                created_at=created_at.isoformat(),
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
            )
        )
        self.buffer_storage.push(request)
        return
    
    def update_generation(
        self,
        observation_id: str,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        request = UpdateGenerationRequest(
            body=UpdateGenerationBody(
                observation_id=observation_id,
                ended_at=ended_at.isoformat() if ended_at else None,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
            )
        )
        self.buffer_storage.push(request)
        return

    def send_requests(
        self,
        requests: List[
            UnionRequest
        ],
    ):
        """
        Sends a batch of requests to the API endpoint.

        Args:
            requests (List[Union[OpenSessionBody, CaptureRecordBody, CaptureObservationBody]]):
                A list of requests to be sent to the API endpoint.

        Returns:
            None: This method does not return any value.

        Raises:
            Exception: If an error occurs while sending the requests.

        """
        # logger.info(requests)
        for attempt in range(self.max_retry):
            logger.info(requests)
            try:
                response = self.api_client.execute(
                    self.api_key,
                    self.endpoint,
                    "/batch",
                    method="POST",
                    json={"batch": [request.model_dump() for request in requests]},
                    timeout=10,
                )
                if response.status_code == 200:
                    return
            except Exception as e:
                print(e)
                time.sleep(2**attempt)
                continue

    def consume_buffer(self) -> None:
        """
        Continuously consumes the buffer and sends requests to the API.

        This method runs in a loop until the `_running` flag is set to False. It checks the buffer size and waits for
        the specified flush interval if the buffer is empty. If the buffer has data, it pulls a batch of data from
        the buffer and sends requests to the API using the `send_requests` method. The completion of the API task is
        handled by the `_handle_api_task_completion` method.

        Raises:
            Exception: If an error occurs while consuming the buffer.

        """
        while self._running:
            try:
                with self.buffer_storage.buffer_lock:
                    if self.buffer_storage.buffer_size == 0:
                        self.buffer_storage.not_empty_cv.wait(self.flush_interval)
                        continue
                    out = self.buffer_storage.pull(self.flush_batch_size)
                    future = self.api_pool.submit(self.send_requests, out)
                    future.add_done_callback(self._handle_api_task_completion)
                    self.buffer_storage.not_empty_cv.wait(self.flush_interval)
            except Exception as e:
                print(e)

    def flush(self) -> None:
        """
        Flushes the buffer by sending the buffered requests.

        This method pulls all the requests from the buffer and sends them in batches
        of size `self.flush_batch_size` to the `send_requests` method.

        Raises:
            Exception: If an error occurs while flushing the buffer.
        """
        try:
            with self.buffer_storage.buffer_lock:
                if self.buffer_storage.buffer_size == 0:
                    return
                out = self.buffer_storage.pull_all()
                # for request chunks size of self.flush_batch_size
                for i in range(0, len(out), self.flush_batch_size):
                    request = out[i : i + self.flush_batch_size]
                    self.send_requests(request)
        except Exception as e:
            print(e)

    def stop(self) -> None:
        """
        Stops the worker thread and flushes any remaining data in the buffer.

        This method sets the `_running` flag to False, notifies the buffer storage
        that it is not empty, joins the worker thread, and then calls the `flush`
        method to ensure any remaining data in the buffer is processed.

        Returns:
            None
        """
        self._running = False
        with self.buffer_storage.buffer_lock:
            self.buffer_storage.not_empty_cv.notify()
        self._thread.join()
        self.flush()

    def _handle_api_task_completion(self, future: Future) -> None:
        try:
            future.result()
        except Exception as e:
            print(e)
