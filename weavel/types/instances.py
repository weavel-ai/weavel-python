from typing import Dict, Optional, Any, Literal
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import Field

from weavel._types import WeavelObject
from weavel._worker import Worker as WeavelWorker

class Session(WeavelObject):
    """Session object."""
    user_id: str
    session_id: str
    created_at: datetime
    metadata: Optional[Dict[str, str]] = None
    weavel_client: WeavelWorker
    
    def log(
        self,
        type: Literal["message", "trace", "track_event"],
        log_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        reason_log_id: Optional[str] = None,
        role: Optional[Literal["user", "assistant", "system"]] = None,
        content: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
    ):
        if log_id is None:
            log_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        
        if type == "message":
            if role is None or content is None:
                raise ValueError("role and content must be provided for message logs.")
            message = Message(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=log_id,
                created_at=created_at,
                metadata=metadata,
                reason_log_id=reason_log_id,
                role=role,
                content=content,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_log(
                type="message",
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=log_id,
                created_at=message.created_at,
                metadata=metadata,
                reason_log_id=reason_log_id,
                role=role,
                content=content,
            )
            return message
    
        elif type == "trace":
            if content is None:
                raise ValueError("content must be provided for trace logs.")
            trace = Trace(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=log_id,
                created_at=created_at,
                metadata=metadata,
                reason_log_id=reason_log_id,
                content=content,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_log(
                type="trace",
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=log_id,
                created_at=trace.created_at,
                metadata=metadata,
                reason_log_id=reason_log_id,
                content=content,
            )
            return trace
        
        elif type == "track_event":
            if content is None:
                raise ValueError("content must be provided for track_event logs.")
            track_event = TrackEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=log_id,
                created_at=created_at,
                metadata=metadata,
                reason_log_id=reason_log_id,
                content=content,
                properties=properties,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_log(
                type="track_event",
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=log_id,
                created_at=track_event.created_at,
                metadata=metadata,
                reason_log_id=reason_log_id,
                content=content,
                properties=properties,
            )
            return track_event
        
        else:
            raise ValueError("Invalid log type.")
    
    def message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        log_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        reason_log_id: Optional[str] = None,
    ):
        if log_id is None:
            log_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        message = Message(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=log_id,
            created_at=created_at,
            metadata=metadata,
            reason_log_id=reason_log_id,
            role=role,
            content=content,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_log(
            type="message",
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=log_id,
            created_at=message.created_at,
            metadata=metadata,
            reason_log_id=reason_log_id,
            role=role,
            content=content,
        )
        return message
    
    def trace(
        self,
        content: str,
        log_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        reason_log_id: Optional[str] = None,
    ):
        if log_id is None:
            log_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        trace = Trace(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=log_id,
            created_at=created_at,
            metadata=metadata,
            reason_log_id=reason_log_id,
            content=content,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_log(
            type="trace",
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=log_id,
            created_at=trace.created_at,
            metadata=metadata,
            reason_log_id=reason_log_id,
            content=content,
        )
        return trace
    
    def track_event(
        self,
        content: str,
        log_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        reason_log_id: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
    ):
        if log_id is None:
            log_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        track_event = TrackEvent(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=log_id,
            created_at=created_at,
            metadata=metadata,
            reason_log_id=reason_log_id,
            content=content,
            properties=properties,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_log(
            type="track_event",
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=log_id,
            created_at=track_event.created_at,
            metadata=metadata,
            reason_log_id=reason_log_id,
            content=content,
            properties=properties,
        )
        return track_event

class Log(WeavelObject):
    user_id: str
    session_id: str 
    log_id: str
    created_at: datetime
    metadata: Optional[Dict[str, str]] = None
    reason_log_id: Optional[str] = None
    weavel_client: WeavelWorker
    
    def observation(
        self,
        type: Literal["event", "generation", "span"],
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        if type == "event":
            if name is None:
                raise ValueError("name must be provided for event observations.")
            event = Event(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=created_at,
                metadata=metadata,
                name=name,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_observation(
                type="event",
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=event.created_at,
                metadata=metadata,
                name=name,
            )   
            return event
        
        elif type == "generation":
            if inputs is None or outputs is None:
                raise ValueError("inputs and outputs must be provided for generation observations.")
            generation =  Generation(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=created_at,
                metadata=metadata,
                inputs=inputs,
                outputs=outputs,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_observation(
                type="generation",
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=generation.created_at,
                metadata=metadata,
                inputs=inputs,
                outputs=outputs,
            )
            return generation
        
        elif type == "span":
            if name is None:
                raise ValueError("name must be provided for span observations.")
            span = Span(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=created_at,
                metadata=metadata,
                name=name,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_observation(
                type="span",
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=span.created_at,
                metadata=metadata,
                name=name,
            )
            return span
        
        else:
            raise ValueError("Invalid observation type.")
        
    
    def event(
        self,
        name: str,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        event = Event(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="event",
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=event.created_at,
            metadata=metadata,
            name=name,
        )
        return event
        
    def generation(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        generation = Generation(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            inputs=inputs,
            outputs=outputs,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="generation",
            log_id=self.log_id,
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
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        span = Span(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            name=name,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="span",
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=span.created_at,
            metadata=metadata,
            name=name,
        )
        return span

class Message(Log):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"]
    content: str

class TrackEvent(Log):
    type: Literal["track_event"] = "track_event"
    content: str
    properties: Optional[Dict[str, str]] = None 

class Trace(Log):
    type: Literal["trace"] = "trace"
    content: str


class Observation(WeavelObject):
    user_id: str
    session_id: str
    log_id: str
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, str]] = None
    parent_observation_id: Optional[str] = None
    weavel_client: WeavelWorker

class Event(Observation):
    type: Literal["event"] = "event"
    name: str

class Generation(Observation):
    type: Literal["generation"] = "generation"
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

class Span(Observation):
    type: Literal["span"] = "span"
    name: str
    
    def observation(
        self,
        type: Literal["event", "generation", "span"],
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        if type == "event":
            if name is None:
                raise ValueError("name must be provided for event observations.")
            event = Event(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=created_at,
                metadata=metadata,
                name=name,
                parent_observation_id=self.observation_id,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_observation(
                type="event",
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=event.created_at,
                metadata=metadata,
                name=name,
            )
            return event
        
        elif type == "generation":
            if inputs is None or outputs is None:
                raise ValueError("inputs and outputs must be provided for generation observations.")
            generation = Generation(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=self.log_id,
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
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=generation.created_at,
                metadata=metadata,
                inputs=inputs,
                outputs=outputs,
            )
            return generation
        
        elif type == "span":
            if name is None:
                raise ValueError("name must be provided for span observations.")
            span = Span(
                user_id=self.user_id,
                session_id=self.session_id,
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=created_at,
                metadata=metadata,
                name=name,
                parent_observation_id=self.observation_id,
                weavel_client=self.weavel_client,
            )
            self.weavel_client.capture_observation(
                type="span",
                log_id=self.log_id,
                observation_id=observation_id,
                created_at=span.created_at,
                metadata=metadata,
                name=name,
            )
            return span
        
        else:
            raise ValueError("Invalid observation type.")
    
    def event(
        self,
        name: str,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        event = Event(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="event",
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=event.created_at,
            metadata=metadata,
            name=name,
        )
        return event
    
    def generation(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)   
            
        generation = Generation(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=self.log_id,
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
            log_id=self.log_id,
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
    ):
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)
            
        span = Span(
            user_id=self.user_id,
            session_id=self.session_id,
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_observation(
            type="span",
            log_id=self.log_id,
            observation_id=observation_id,
            created_at=span.created_at,
            metadata=metadata,
            name=name,
        )
        return span