from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal
from enum import Enum
from uuid import UUID
from datetime import datetime, date


class WeavelObject(BaseModel):
    """Weavel Object. Extended Pydantic BaseModel with support for UUID and datetime."""
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data: Any):
        for key, value in data.items():
            if key in self.__annotations__ and (
                self.__annotations__[key] == str
                or self.__annotations__[key] == Optional[str]
            ):
                if (
                    isinstance(value, UUID)
                    or isinstance(value, datetime)
                    or isinstance(value, date)
                ):
                    try:
                        data[key] = str(value)
                    except:
                        data[key] = value
        super().__init__(**data)

class OpenSessionBody(WeavelObject):
    """Start Session body."""

    object_type: Literal["open_session"] = "open_session"
    user_id: str
    session_id: str
    created_at: str
    metadata: Optional[Dict[str, str]] = None

class CaptureLogBody(WeavelObject):
    """Capture Log body."""

    object_type: Literal["capture_log"] = "capture_log"
    type: str
    user_id: str
    session_id: str
    role: Optional[str] = None
    content: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    log_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    reason_log_id: Optional[str] = None
    created_at: str


class CaptureObservationBody(WeavelObject):
    object_type: Literal["capture_observation"] = "capture_observation"
    type: str
    log_id: str
    created_at: str
    name: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    parent_observation_id: Optional[str] = None
    observation_id: Optional[str] = None


class SaveUserIdentityBody(WeavelObject):
    """Save user identity body."""

    object_type: Literal["save_user_identity"] = "save_user_identity"
    user_id: str
    properties: Dict[str, Any]
    created_at: str


class BatchRequest(WeavelObject):
    batch: List[Union[OpenSessionBody, CaptureLogBody, SaveUserIdentityBody]]
