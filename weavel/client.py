from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import pendulum
from dotenv import load_dotenv

from weavel.types import DataType
from weavel._worker import Worker

class WeavelClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self.log = Worker(self.api_key)

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
    ) -> str:
        """Start the new trace for user_uuid.
        
        Args:
            user_uuid: The user's UUID.
        Returns:
            The trace UUID.
        """
        trace_uuid = str(uuid.uuid4())
        self.log._start_trace(trace_uuid, user_uuid, timestamp, metadata)
        return trace_uuid
    
    def add_metadata_to_trace(
        self,
        trace_uuid: str,
        metadata: Dict[str, str],
    ):
        """Add metadata to the trace.
        
        Args:
            trace_uuid: The trace UUID.
            metadata: The metadata.
        """
        self.log._save_trace_metadata(trace_uuid, metadata)
        return
    
    def close(
        self
    ):
        """Close the client."""
        self.log.stop()
    
    
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