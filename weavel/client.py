from __future__ import annotations

import os
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
        user_id: str,
        trace_id: str,
    ):
        self.worker = worker
        self.user_id = user_id
        self.trace_id = trace_id
    
    def log_message(
        self,
        type: str, 
        content: str,
        timestamp: Optional[datetime] = None,
        unit_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        if type == "user":
            self.worker.user_message(self.user_id, self.trace_id, content, unit_name, timestamp, metadata)
        elif type == "assistant":
            self.worker.assistant_message(self.user_id, self.trace_id, content, unit_name, timestamp, metadata)
        elif type == "system":
            self.worker.system_message(self.user_id, self.trace_id, content, unit_name, timestamp, metadata)
        else:
            raise ValueError("Invalid message type.")
    
    def log_inner_step(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
        unit_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        self.worker.inner_step(self.user_id, self.trace_id, content, unit_name, timestamp, metadata)
            
class WeavelClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self._worker = Worker(self.api_key)

    # def create_user_id(
    #     self,
    # ) -> str:
    #     """Create a new user_id.
        
    #     Returns:
    #         The user identifier.
    #     """
    #     return str(uuid.uuid4())
    
    def start_trace(
        self,
        user_id: str,
        trace_id: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Trace:
        """Start the new trace for user_id.
        
        Args:
            user_id: The user's identifier.
            trace_id: The trace identifier.
        Returns:
            The trace instance.
        """
        self._worker._start_trace(trace_id, user_id, timestamp, metadata)
        trace = Trace(self._worker, user_id, trace_id)
        return trace
    
    def resume_trace(
        self,
        user_id,
        trace_id,
    ) -> Trace:
        """Resume the trace for (user_id, trace_id).
        Method for make Trace object for existing trace.
        """
        
        return Trace(self._worker, user_id, trace_id)
    
    def track(
        self,
        user_id: str,
        event_name: str,
        properties: Dict
    ):
        self._worker._track_users(user_id, event_name, properties)
    
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