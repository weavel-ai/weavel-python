from pydantic import BaseModel as PydanticBaseModel, validator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal
from enum import Enum
from uuid import UUID
from datetime import datetime, date


class BaseModel(PydanticBaseModel):
    """Extended Pydantic BaseModel"""
    class Config:
        arbitrary_types_allowed = True
      

class OpenSessionBody(BaseModel):
    """Start Session body."""

    object_type: Literal["open_session"] = "open_session"
    user_id: str
    session_id: str
    created_at: str
    metadata: Optional[Dict[str, str]] = None

class CaptureRecordBody(BaseModel):
    """Capture Log body."""

    object_type: Literal["capture_record"] = "capture_record"
    type: str
    user_id: str
    session_id: str
    role: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    record_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    reason_record_id: Optional[str] = None
    created_at: str


class CaptureObservationBody(BaseModel):
    object_type: Literal["capture_observation"] = "capture_observation"
    type: str
    record_id: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    value: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    parent_observation_id: Optional[str] = None
    observation_id: Optional[str] = None


class SaveUserIdentityBody(BaseModel):
    """Save user identity body."""

    object_type: Literal["save_user_identity"] = "save_user_identity"
    user_id: str
    properties: Dict[str, Any]
    created_at: str


class BatchRequest(BaseModel):
    batch: List[Union[OpenSessionBody, CaptureRecordBody, SaveUserIdentityBody]]
