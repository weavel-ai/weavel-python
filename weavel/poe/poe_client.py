from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import pendulum
from dotenv import load_dotenv
from fastapi_poe import QueryRequest, PartialResponse

from weavel.types import DataType
from weavel.poe._poe_worker import PoeWorker

load_dotenv()

class WeavelPoeClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self.worker = PoeWorker(self.api_key)
        
    def log(
        self,
        user_request: QueryRequest,
        bot_responses: List[PartialResponse],
        response_timestamp: Optional[datetime] = datetime.now().isoformat(),
    ):
        self.worker.log(user_request, bot_responses, response_timestamp)
        return
    
    def close(
        self
    ):
        """Close the client."""
        self.worker.stop()
    
    
def create_poe_client(
    api_key: Optional[str] = None,
) -> WeavelPoeClient:
    """Create a Weavel client for Poe integration.
    
    Args:
        api_key: The API key.
    Returns:
        The Weavel client.
    """
    return WeavelPoeClient(api_key=api_key)