from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import pendulum

from weavel.types import DataType

class WeavelClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."

    def start_trace(
        self,
        user_uuid: str
    ) -> str:
        """Start the new trace for user_uuid.
        
        Args:
            user_uuid: The user's UUID.
        Returns:
            The trace UUID.
        """
        pass
    
    def log(
        self,
        trace_uuid: str,
        data_type: DataType,
        data_content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None, 
    ):
        """Log the data to the trace.
        
        Args:
            trace_uuid: The trace UUID.
            data_type: The data type.
            data_content: The data content.
            timestamp: The timestamp.
            metadata: The metadata.
        """
        pass