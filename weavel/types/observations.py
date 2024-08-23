from typing import Dict, List, Optional, Any, Literal, Union
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import Field

from weavel._request import BaseModel


class Observation(BaseModel):
    record_id: Optional[str] = None
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    parent_observation_id: Optional[str] = None


class Span(Observation):
    type: Literal["span"] = "span"
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    ended_at: Optional[datetime] = None


class Generation(Observation):
    type: Literal["generation"] = "generation"
    prompt_name: Optional[str] = None
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    ended_at: Optional[datetime] = None


class Log(Observation):
    type: Literal["log"] = "log"
    value: Optional[str] = None
