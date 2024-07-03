from typing import Dict, Optional, Any, Literal
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import Field

from weavel._types import BaseModel
from weavel._worker import Worker as WeavelWorker


class Observation(BaseModel):
    user_id: str
    session_id: str
    record_id: str
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, str]] = None
    parent_observation_id: Optional[str] = None
    weavel_client: WeavelWorker

class Log(Observation):
    type: Literal["log"] = "log"
    name: str
    value: Optional[str] = None

class Generation(Observation):
    type: Literal["generation"] = "generation"
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

class Span(Observation):
    type: Literal["span"] = "span"
    name: str
    
    def log(
        self,
        name: str,
        value: Optional[str] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Log:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        log = Log(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            value=value,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="log",
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=log.created_at,
            metadata=metadata,
            name=name,
            value=value,
            parent_observation_id=self.observation_id,
        )
        return log
    
    def generation(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Generation:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)   
            
        generation = Generation(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            inputs=inputs,
            outputs=outputs,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="generation",
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=generation.created_at,
            metadata=metadata,
            inputs=inputs,
            outputs=outputs,
            parent_observation_id=self.observation_id,
        )
        return generation
    
    def span(
        self,
        name: str,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Span":
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        span = Span(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="span",
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=span.created_at,
            metadata=metadata,
            name=name,
            parent_observation_id=self.observation_id,
        )
        return span


class Record(BaseModel):
    user_id: str
    session_id: str 
    record_id: str
    created_at: datetime
    metadata: Optional[Dict[str, str]] = None
    reason_record_id: Optional[str] = None
    weavel_client: WeavelWorker


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
    name: str
    
    def log(
        self,
        name: str,
        value: Optional[str] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Log:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        log = Log(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            value=value,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="log",
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=log.created_at,
            metadata=metadata,
            name=name,
            value=value,
        )
        return log
        
    def generation(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Generation:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        generation = Generation(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            inputs=inputs,
            outputs=outputs,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="generation",
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=generation.created_at,
            metadata=metadata,
            inputs=inputs,
            outputs=outputs,
        )
        return generation
        
    def span(
        self,
        name: str,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> Span:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        span = Span(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            name=name,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="span",
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=span.created_at,
            metadata=metadata,
            name=name,
        )
        return span


class Session(BaseModel):
    """Session object."""
    user_id: str
    session_id: str
    created_at: datetime
    metadata: Optional[Dict[str, str]] = None
    weavel_client: WeavelWorker
    
    def message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        record_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        reason_record_id: Optional[str] = None,
    ) -> Message:
        if record_id is None:
            record_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        message = Message(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=record_id,
            created_at=created_at,
            metadata=metadata,
            reason_record_id=reason_record_id,
            role=role,
            content=content,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_record(
            type="message",
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=record_id,
            created_at=message.created_at,
            metadata=metadata,
            reason_record_id=reason_record_id,
            role=role,
            content=content,
        )
        return message
    
    def trace(
        self,
        name: str,
        record_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        reason_record_id: Optional[str] = None,
    ) -> Trace:
        if record_id is None:
            record_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        trace = Trace(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=record_id,
            created_at=created_at,
            metadata=metadata,
            reason_record_id=reason_record_id,
            name=name,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_record(
            type="trace",
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=record_id,
            created_at=trace.created_at,
            metadata=metadata,
            reason_record_id=reason_record_id,
            name=name,
        )
        return trace
    
    def track_event(
        self,
        name: str,
        properties: Optional[Dict[str, str]] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        record_id: Optional[str] = None,
        reason_record_id: Optional[str] = None,
    ) -> TrackEvent:
        if record_id is None:
            record_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        track_event = TrackEvent(
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=record_id,
            created_at=created_at,
            metadata=metadata,
            reason_record_id=reason_record_id,
            name=name,
            properties=properties,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_record(
            type="track_event",
            user_id=self.user_id,
            session_id=self.session_id,
            record_id=record_id,
            created_at=track_event.created_at,
            metadata=metadata,
            reason_record_id=reason_record_id,
            name=name,
            properties=properties,
        )
        return track_event
