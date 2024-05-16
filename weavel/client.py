from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Literal, Optional, Any

from dotenv import load_dotenv

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
        # pylint: disable=redefined-builtin
        type: Literal["user", "assistant"],
        content: str,
        timestamp: Optional[datetime] = None,
        unit_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Logs a message with the specified type, content, timestamp, unit name, and metadata.

        Args:
            type (Literal["user", "assistant"]): The type of the message. Must be either "user" or "assistant".
            content (str): The content of the message.
            timestamp (Optional[datetime], optional): The timestamp of the message. Defaults to None.
            unit_name (Optional[str], optional): The unit name associated with the message. Defaults to None.
            metadata (Optional[Dict[str, str]], optional): Additional metadata for the message. Defaults to None.

        Raises:
            ValueError: If an invalid message type is provided.
        """
        if type == "user":
            self.worker.log_user_message(
                self.user_id, self.trace_id, content, unit_name, timestamp, metadata
            )
        elif type == "assistant":
            self.worker.log_assistant_message(
                self.user_id, self.trace_id, content, unit_name, timestamp, metadata
            )
        else:
            raise ValueError("Invalid message type.")

    def log_inner_step(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
        unit_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Logs an inner step in the worker.

        Args:
            content (str): The content of the inner step.
            timestamp (Optional[datetime], optional): The timestamp of the inner step. Defaults to None.
            unit_name (Optional[str], optional): The unit name of the inner step. Defaults to None.
            metadata (Optional[Dict[str, str]], optional): Additional metadata for the inner step. Defaults to None.
        """
        self.worker.log_inner_step(
            self.user_id, self.trace_id, content, unit_name, timestamp, metadata
        )


class WeavelClient:
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

    def open_trace(
        self,
        user_id: str,
        trace_id: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Trace:
        """Start a new trace for the specified user.

        Args:
            user_id (str): The user's identifier.
            trace_id (str): The trace identifier.
            timestamp (datetime, optional): The timestamp for the trace. If not provided,
                the current timestamp will be used.
            metadata (Dict[str, str], optional): Additional metadata for the trace.

        Returns:
            Trace: The trace instance.

        """
        self._worker.open_trace(trace_id, user_id, timestamp, metadata)
        trace = Trace(self._worker, user_id, trace_id)
        return trace

    def resume_trace(
        self,
        user_id: str,
        trace_id: str,
    ) -> Trace:
        """Resume an existing trace for the specified user.

        Args:
            user_id (str): The user's identifier.
            trace_id (str): The trace identifier.

        Returns:
            Trace: The trace instance.

        """
        return Trace(self._worker, user_id, trace_id)

    def track(self, user_id: str, event_name: str, properties: Dict, trace_id: Optional[str] = None):
        """Track a user's track event.

        This method is used to track user actions such as "paid", "subscribed", "unsubscribed", etc.

        Args:
            user_id (str): The identifier of the user.
            event_name (str): The name of the track event.
            properties (Dict): The properties of the track event.
            trace_id (str, optional): The ID of the trace associated with the track event.

        """
        self._worker.log_track_event(user_id, event_name, properties, trace_id)
        
    def identify(self, user_id: str, properties: Dict[str, Any]):
        """Identify a user with the specified properties.
        
        You can save any user information you know about a user for user_id, such as their email, name, or phone number in dict format.
        Properties will be updated for every call.
        """
        
        self._worker.identify_user(user_id, properties)
        

    def close(self):
        """Close the client connection."""
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
