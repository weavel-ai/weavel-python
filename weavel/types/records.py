from typing import Dict, Optional, Any, Literal
from datetime import datetime

from weavel._request import BaseModel


class Record(BaseModel):
    session_id: Optional[str] = None
    record_id: str
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None
    ref_record_id: Optional[str] = None


class Message(Record):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"]
    content: str


class TrackEvent(Record):
    type: Literal["track_event"] = "track_event"
    name: str
    properties: Optional[Dict[str, str]] = None


class Trace(Record):
    type: Literal["trace"] = "trace"
    name: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    ended_at: Optional[datetime] = None
