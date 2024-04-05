from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal
from enum import Enum
from uuid import UUID
from datetime import datetime, date


class WeavelObject(BaseModel):
    """Weavel Object. Extended Pydantic BaseModel with support for UUID and datetime."""

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


class TraceDataRole(str, Enum):
    system = "system"
    user = "user"
    assisatant = "assistant"
    inner_step = "inner_step"
    # retrieved_content = "retrieved_content"


class BackgroundTaskType(str, Enum):
    open_trace = "open_trace"
    capture_trace_data = "capture_trace_data"
    capture_track_event = "capture_track_event"
    create_semantic_event = "create_semantic_event"
    extract_keywords = "extract_keywords"
    save_user_identity = "save_user_identity"


class OpenTraceBody(WeavelObject):
    """Start Trace body."""

    type: Literal["open_trace"] = "open_trace"
    user_id: str
    trace_id: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class CaptureTrackEventBody(WeavelObject):
    """Capture track_event body."""

    type: Literal["capture_track_event"] = "capture_track_event"
    user_id: str
    track_event_name: str
    properties: Dict[str, Any]
    timestamp: Optional[str] = None


class CaptureTraceDataBody(WeavelObject):
    """Capture Trace Data body."""

    type: Literal["capture_trace_data"] = "capture_trace_data"
    user_id: str
    trace_id: str
    role: str
    content: str
    unit_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class SaveUserIdentityBody(WeavelObject):
    """Save user identity body."""
    type: Literal["save_user_identity"] = "save_user_identity"
    user_id: str
    properties: Dict[str, Any]
    timestamp: Optional[str] = None

# class SaveMetadataTraceBody(WeavelObject):
#     """Save metadata body."""
#     trace_id: str
#     metadata: Dict[str, str]


class BatchRequest(WeavelObject):
    batch: List[Union[OpenTraceBody, CaptureTrackEventBody, CaptureTraceDataBody]]
