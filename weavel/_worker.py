from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from threading import Thread
from concurrent.futures import Future, ThreadPoolExecutor

from weavel._types import (
    OpenSessionBody,
    CaptureRecordBody,
    CaptureObservationBody,
    SaveUserIdentityBody
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
        user_id: str,
        session_id: str,
        created_at: datetime,
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
        request = OpenSessionBody(
            user_id=user_id,
            session_id=session_id,
            created_at=created_at.isoformat(),
            metadata=metadata,
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
        request = SaveUserIdentityBody(
            user_id=user_id,
            properties=properties,
            created_at=str(datetime.now(timezone.utc).isoformat()),
        )
        self.buffer_storage.push(request)

        return
    
    def capture_record(
        self,
        type: str,
        user_id: str,
        session_id: str,
        record_id: str,
        created_at: datetime,
        metadata: Optional[Dict[str, str]] = None,
        reason_record_id: Optional[str] = None,
        role: Optional[str] = None,
        content: Optional[str] = None,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Log "Log" type data to the session.

        Args:
            session_id: The session identifier.
            user_id: The user identifier.
            type: The type of log.
            created_at: The created_at of the log.
            role: The role of the log.
            content: The content of the log.
            properties: The properties of the log.
            record_id: The log identifier.
            metadata: The metadata of the log.
            reason_record_id: The reason log identifier.
        """
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = created_at
            
        request = CaptureRecordBody(
            user_id=user_id,
            session_id=session_id,
            type=type,
            role=role,
            content=content,
            name=name,
            properties=properties,
            created_at=created_at.isoformat(),
            record_id=record_id,
            metadata=metadata,
            reason_record_id=reason_record_id,
        )
        self.buffer_storage.push(request)
        return

    def capture_observation(
        self,
        type: str,
        record_id: str,
        observation_id: str,
        created_at: datetime,
        metadata: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        value: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        if created_at.tzinfo is None or created_at.tzinfo.utcoffset(created_at) is None:
            created_at = created_at.astimezone(timezone.utc)
        else:
            created_at = created_at
        
        request = CaptureObservationBody(
            type=type,
            record_id=record_id,
            created_at=created_at.isoformat(),
            metadata=metadata,
            name=name,
            value=value,
            inputs=inputs,
            outputs=outputs,
            parent_observation_id=parent_observation_id,
            observation_id=observation_id,
        )
        self.buffer_storage.push(request)
        return

    def send_requests(
        self,
        requests: List[
            Union[OpenSessionBody, CaptureRecordBody, CaptureObservationBody]
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
