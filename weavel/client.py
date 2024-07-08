from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Literal, Optional, Any
from uuid import uuid4

from dotenv import load_dotenv

from weavel._worker import Worker
from weavel.types.instances import Session, Span, Trace

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
        user_id: Optional[str] = None,
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
        if user_id is None and session_id is None:
            raise ValueError("user_id or session_id must be provided.")
        
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
        if user_id is not None:
            self._worker.open_session(
                session_id=session_id,
                created_at=created_at,
                user_id=user_id,
                metadata=metadata
            )
        return session
        
    def identify(self, user_id: str, properties: Dict[str, Any]):
        """Identify a user with the specified properties.
        
        You can save any user information you know about a user for user_id, such as their email, name, or phone number in dict format.
        Properties will be updated for every call.
        """
        
        self._worker.identify_user(user_id, properties)
        
    def trace(
        self,
        session_id: Optional[str] = None,
        record_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ref_record_id: Optional[str] = None,
    ) -> Trace:
        """Create a new trace record or fetch an existing one

        Args:
            session_id (str, optional): The session ID for the trace.
            record_id (str, optional): The record ID. If not provided, a new record ID will be generated.
            created_at (datetime, optional): The created_at for the trace. If not provided, the current time will be used.
            name (str, optional): The name of the trace.
            inputs (Dict[str, Any], optional): The inputs for the trace.
            outputs (Dict[str, Any], optional): The outputs for the trace.
            metadata (Dict[str, Any], optional): Additional metadata for the trace.
            ref_record_id (str, optional): The record ID to reference.

        """
        if session_id is None and record_id is None:
            raise ValueError("session_id or record_id must be provided.")

        if session_id is not None and name is None:
            raise ValueError("If you want to create a new trace, you must provide a name.")
        
        if session_id is not None:
            # Create a new trace
            if record_id is None:
                record_id = str(uuid4())
            if created_at is None:
                created_at = datetime.now(timezone.utc)
            self._worker.capture_trace(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at,
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                ref_record_id=ref_record_id
            )
            return Trace(
                session_id=session_id,
                record_id=record_id,
                created_at=created_at,
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                weavel_client=self._worker
            )
        else:
            # Fetch an existing trace
            return Trace(
                record_id=record_id,
                weavel_client=self._worker
            )
    
    def span(
        self,
        record_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
    ):
        if record_id is None and observation_id is None:
            raise ValueError("record_id or observation_id must be provided.")

        if record_id is not None and name is None:
            raise ValueError("If you want to create a new span, you must provide a name.")

        if record_id is not None:
            # Create a new span
            if observation_id is None:
                observation_id = str(uuid4())
            if created_at is None:
                created_at = datetime.now(timezone.utc)
            self._worker.capture_span(
                record_id=record_id,
                observation_id=observation_id,
                created_at=created_at,
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id
            )
            return Span(
                record_id=record_id,
                observation_id=observation_id,
                created_at=created_at,
                name=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
                weavel_client=self._worker
            )
        else:
            # Fetch an existing span
            return Span(
                observation_id=observation_id,
                weavel_client=self._worker
            )
    
    def track(
        self,
        session_id: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
        ref_record_id: Optional[str] = None,
    ) -> None:
        """Track an event for the specified session.

        Args:
            session_id (str): The session ID for the event.
            name (str): The name of the event.
            properties (Dict[str, Any], optional): Additional properties for the event.

        """
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        
        self._worker.capture_track_event(
            session_id=session_id,
            record_id=record_id,
            created_at=created_at,
            name=name,
            properties=properties,
            metadata=metadata,
            ref_record_id=ref_record_id
        )

    def close(self):
        """Close the client connection."""
        self._worker.stop()
    
    def flush(self):
        """Flush the buffer."""
        self._worker.flush()

