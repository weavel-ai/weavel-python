from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from threading import Thread
from concurrent.futures import Future, ThreadPoolExecutor

from weavel.types import (
    DataType,
    BackgroundTaskType,
    WeavelRequest,
    StartTraceBody,
    SaveTraceDataBody,
    SaveTrackEventBody
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
        if not hasattr(self, 'is_initialized'):
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
        
    def _start_trace(self, trace_id: str, user_id: str, timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, str]] = None) -> str:
        """Start the new trace for user_id.
        
        Returns:
            The trace id.
        """
        # add task to buffer
        request = WeavelRequest(**{
            "task": BackgroundTaskType.start_trace.value,
            "body" : StartTraceBody(**{
                "timestamp": str(timestamp or datetime.now().isoformat()),
                "trace_id": trace_id,
                "user_id": user_id,
                "metadata": metadata,
            })
        })
        self.buffer_storage.push(request)
        
        return trace_id
    
    def _track_users(self, user_id: str, name: str, properties: Dict) -> None:
        """Save the user event"""
        request = WeavelRequest(**{
            "task": BackgroundTaskType.log_track_event.value,
            "body" : SaveTrackEventBody(**{
                "timestamp": str(datetime.now().isoformat()),
                "user_id": user_id,
                "track_event_name": name,
                "properties": properties,
            })
        })
        self.buffer_storage.push(request)
        
        return
        
    def user_message(
        self,
        user_id: str,
        trace_id: str,
        data_content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None, 
    ):
        """Log the "user_message" type data to the trace.
        
        Args:
            trace_id: The trace identifier.
            data_content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = WeavelRequest(**{
            "task": BackgroundTaskType.log_trace_data.value,
            "body" : SaveTraceDataBody(**{
                "timestamp": str(timestamp or datetime.now().isoformat()),
                "user_id": user_id,
                "trace_id": trace_id,
                "data_type": DataType.user_message,
                "data_content": data_content,
                "unit_name": unit_name,
                "metadata": metadata,
            })
        })
        self.buffer_storage.push(request)
        return
    
    def assistant_message(
        self,
        user_id: str,
        trace_id: str,
        data_content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Log the "llm_background_message" type data to the trace.
        
        Args:
            trace_id: The trace identifier.
            data_content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = WeavelRequest(**{
            "task": BackgroundTaskType.log_trace_data.value,
            "body" : SaveTraceDataBody(**{
                "timestamp": str(timestamp or datetime.now().isoformat()),
                "user_id": user_id,
                "trace_id": trace_id,
                "data_type": DataType.assistant_message,
                "data_content": data_content,
                "unit_name": unit_name,
                "metadata": metadata,
            })
        })
        self.buffer_storage.push(request)
        return
    
    def system_message(
        self,
        user_id: str,
        trace_id: str,
        data_content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Log the "system_message" type data to the trace.
        
        Args:
            trace_id: The trace identifier.
            data_content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = WeavelRequest(**{
            "task": BackgroundTaskType.log_trace_data.value,
            "body" : SaveTraceDataBody(**{
                "timestamp": str(timestamp or datetime.now().isoformat()),
                "user_id": user_id,
                "trace_id": trace_id,
                "data_type": DataType.system_message,
                "data_content": data_content,
                "unit_name": unit_name,
                "metadata": metadata,
            })
        })
        self.buffer_storage.push(request)
        return
    
    def inner_step(
        self,
        user_id: str,
        trace_id: str,
        data_content: str,
        unit_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Log the "inner_step" type data to the trace.
        
        Args:
            trace_id: The trace identifier.
            data_content: The data content.
            unit_name: The unit name.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        request = WeavelRequest(**{
            "task": BackgroundTaskType.log_trace_data.value,
            "body" : SaveTraceDataBody(**{
                "timestamp": str(timestamp or datetime.now().isoformat()),
                "user_id": user_id,
                "trace_id": trace_id,
                "data_type": DataType.inner_step,
                "data_content": data_content,
                "unit_name": unit_name,
                "metadata": metadata,
            })
        })
        self.buffer_storage.push(request)
        return
    
    def send_request(self, request: WeavelRequest):
        for attempt in range(self.max_retry):
            try:
                response = self.api_client.execute(
                    self.api_key,
                    self.endpoint,
                    "/trace",
                    method="POST",
                    json=request.model_dump(),
                    timeout=10,
                )
                if response.status_code == 200:
                    return
            except Exception as e:
                print(e)
                time.sleep(2**attempt)
                continue
            
    def send_requests(self, requests: List[WeavelRequest]):
        logger.info(requests)
        for attempt in range(self.max_retry):
            try:
                response = self.api_client.execute(
                    self.api_key,
                    self.endpoint,
                    "/trace/batch",
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
            print(e)