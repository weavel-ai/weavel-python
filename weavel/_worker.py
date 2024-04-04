from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from threading import Thread
from concurrent.futures import Future, ThreadPoolExecutor

from weavel.types import (
    TraceDataRole,
    OpenTraceBody,
    CaptureTraceDataBody,
    CaptureTrackEventBody,
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

    def open_trace(
        self,
        trace_id: str,
        user_id: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start the new trace for user_id.

        Returns:
            The trace id.
        """
        # add task to buffer
        request = OpenTraceBody(
            user_id=user_id,
            trace_id=trace_id,
            timestamp=str(timestamp or datetime.now().isoformat()),
            metadata=metadata,
        )

        self.buffer_storage.push(request)

        return trace_id

    def log_track_event(self, user_id: str, name: str, properties: Dict) -> None:
        """
        Logs a track event for a user.

        Args:
            user_id (str): The ID of the user.
            name (str): The name of the track event.
            properties (Dict): Additional properties associated with the track event.

        Returns:
            None
        """
        request = CaptureTrackEventBody(
            user_id=user_id,
            track_event_name=name,
            properties=properties,
            timestamp=str(datetime.now().isoformat()),
        )
        self.buffer_storage.push(request)

        return

    def log_user_message(
        self,
        user_id: str,
        trace_id: str,
        content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Log the "user_message" type data to the trace.

        Args:
            trace_id: The trace identifier.
            content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = CaptureTraceDataBody(
            user_id=user_id,
            trace_id=trace_id,
            role=TraceDataRole.user,
            content=content,
            unit_name=unit_name,
            metadata=metadata,
            timestamp=str(timestamp or datetime.now().isoformat()),
        )
        self.buffer_storage.push(request)
        return

    def log_assistant_message(
        self,
        user_id: str,
        trace_id: str,
        content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Log the "llm_background_message" type data to the trace.

        Args:
            trace_id: The trace identifier.
            content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = CaptureTraceDataBody(
            user_id=user_id,
            trace_id=trace_id,
            role=TraceDataRole.assisatant,
            content=content,
            unit_name=unit_name,
            metadata=metadata,
            timestamp=str(timestamp or datetime.now().isoformat()),
        )
        self.buffer_storage.push(request)
        return

    def log_inner_step(
        self,
        user_id: str,
        trace_id: str,
        content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Log the "inner_step" type data to the trace.

        Args:
            trace_id: The trace identifier.
            content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = CaptureTraceDataBody(
            user_id=user_id,
            trace_id=trace_id,
            role=TraceDataRole.inner_step,
            content=content,
            unit_name=unit_name,
            metadata=metadata,
            timestamp=str(timestamp or datetime.now().isoformat()),
        )
        self.buffer_storage.push(request)
        return

    def send_requests(
        self,
        requests: List[
            Union[OpenTraceBody, CaptureTrackEventBody, CaptureTraceDataBody]
        ],
    ):
        """
        Sends a batch of requests to the API endpoint.

        Args:
            requests (List[Union[OpenTraceBody, CaptureTrackEventBody, CaptureTraceDataBody]]):
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
