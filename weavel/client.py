from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Literal, Optional, Any
from uuid import uuid4

from dotenv import load_dotenv

from weavel._worker import Worker
from weavel.types.instances import Session

load_dotenv()

class Weavel:
    """Client for interacting with the Weavel service.

    This class provides methods for creating and managing traces, tracking user actions,
    and closing the client connection.

    Args:
        api_key (str, optional): The API key for authenticating with the Weavel service.
            If not provided, the API key will be retrieved from the environment variable
            WEAVEL_API_KEY.

    Attributes:
        api_key (str): The API key used for authentication.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("WEAVEL_API_KEY")
        assert self.api_key is not None, "API key not provided."
        self._worker = Worker(self.api_key)
        
    def session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Session:
        """Create a new session for the specified user.

        Args:
            user_id (str): The user ID for the session.
            session_id (str, optional): The session ID. If not provided, a new session ID will be generated.
            created_at (datetime, optional): The created_at for the session. If not provided, the current time will be used.
            metadata (Dict[str, str], optional): Additional metadata for the session.

        Returns:
            Session: The session object.

        """
        if session_id is None:
            session_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        session = Session(
            user_id=user_id,
            session_id=session_id,
            created_at=created_at,
            metadata=metadata,
            weavel_client=self._worker
        )
        self._worker.open_session(
            session.user_id, 
            session.session_id, 
            session.created_at, 
            session.metadata
        )
        return session
        
    def identify(self, user_id: str, properties: Dict[str, Any]):
        """Identify a user with the specified properties.
        
        You can save any user information you know about a user for user_id, such as their email, name, or phone number in dict format.
        Properties will be updated for every call.
        """
        
        self._worker.identify_user(user_id, properties)

    def close(self):
        """Close the client connection."""
        self._worker.stop()
    
    def flush(self):
        """Flush the buffer."""
        self._worker.flush()

