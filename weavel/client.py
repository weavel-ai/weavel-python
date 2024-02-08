from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import pendulum
from dotenv import load_dotenv

from weavel.types import DataType
from weavel._worker import Worker

load_dotenv()

class Trace:
    def __init__(
        self,
        worker: Worker,
        user_uuid: str,
        trace_uuid: str,
    ):
        self.worker = worker
        self.user_uuid = user_uuid
        self.trace_uuid = trace_uuid
    
    def log_message(
        self,
        type: str, 
        content: str,
        timestamp: Optional[datetime] = None,
        unit_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        if type == "user":
            self.worker.user_message(self.user_uuid, self.trace_uuid, content, unit_name, timestamp, metadata)
        elif type == "assistant":
            self.worker.assistant_message(self.user_uuid, self.trace_uuid, content, unit_name, timestamp, metadata)
        elif type == "system":
            self.worker.system_message(self.user_uuid, self.trace_uuid, content, unit_name, timestamp, metadata)
        else:
            raise ValueError("Invalid message type.")
    
    def log_inner_step(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
        unit_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        self.worker.inner_step(self.user_uuid, self.trace_uuid, content, unit_name, timestamp, metadata)
            
class WeavelClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self._worker = Worker(self.api_key)

    def create_user_uuid(
        self,
    ) -> str:
        """Create a new user_uuid.
        
        Returns:
            The user UUID.
        """
        return str(uuid.uuid4())
    
    def start_trace(
        self,
        user_uuid: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Trace:
        """Start the new trace for user_uuid.
        
        Args:
            user_uuid: The user's UUID.
        Returns:
            The trace UUID.
        """
        trace_uuid = str(uuid.uuid4())
        self._worker._start_trace(trace_uuid, user_uuid, timestamp, metadata)
        trace = Trace(self._worker, user_uuid, trace_uuid)
        return trace
    
    def track(
        self,
        user_uuid: str,
        event_name: str,
        properties: Dict
    ):
        self._worker._track_users(user_uuid, event_name, properties)
    
    def close(
        self
    ):
        """Close the client."""
        self._worker.stop()
    
    
def create_client(
    api_key: Optional[str] = None,
) -> WeavelClient:
    """Create a Weavel client.
    
    Args:
        api_key: The API key.
    Returns:
        The Weavel client.
    """
    return WeavelClient(api_key=api_key)