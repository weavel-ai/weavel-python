from typing import Dict, List, Optional, Any, Literal, Union
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import Field

from weavel._request import BaseModel
from weavel._worker import Worker as WeavelWorker


class Observation(BaseModel):
    record_id: Optional[str] = None
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    parent_observation_id: Optional[str] = None
    weavel_client: WeavelWorker


class Log(Observation):
    type: Literal["log"] = "log"
    value: Optional[str] = None


class Generation(Observation):
    type: Literal["generation"] = "generation"
    inputs: Optional[Union[Dict[str, Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], str]] = None
    ended_at: Optional[datetime] = None

    def end(
        self,
        ended_at: Optional[datetime] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self.weavel_client.update_generation(
            observation_id=self.observation_id,
            ended_at=ended_at,
        )
        self.ended_at = ended_at
        return

    def update(
        self,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self.weavel_client.update_generation(
            observation_id=self.observation_id,
            ended_at=ended_at,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        self.ended_at = ended_at
        return


class Span(Observation):
    type: Literal["span"] = "span"
    inputs: Optional[Union[Dict[str, Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], str]] = None
    ended_at: Optional[datetime] = None

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
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            value=value,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_log(
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
        name: str,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Generation:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        generation = Generation(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            inputs=inputs,
            outputs=outputs,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_generation(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=generation.created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=self.observation_id,
        )
        return generation

    def span(
        self,
        name: str,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Span":
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        span = Span(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=self.observation_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_span(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=span.created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=self.observation_id,
        )
        return span

    def end(
        self,
        ended_at: Optional[datetime] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self.weavel_client.update_span(
            observation_id=self.observation_id,
            ended_at=ended_at,
        )
        self.ended_at = ended_at
        return

    def update(
        self,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self.weavel_client.update_span(
            observation_id=self.observation_id,
            ended_at=ended_at,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        self.ended_at = ended_at
        return


class Record(BaseModel):
    session_id: Optional[str] = None
    record_id: str
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None
    ref_record_id: Optional[str] = None
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
    name: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    ended_at: Optional[datetime] = None

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
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            metadata=metadata,
            name=name,
            value=value,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_log(
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
        name: str,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Generation:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        generation = Generation(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_generation(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=generation.created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        return generation

    def span(
        self,
        name: str,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Span:
        if observation_id is None:
            observation_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        span = Span(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_span(
            record_id=self.record_id,
            observation_id=observation_id,
            created_at=span.created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        return span

    def end(
        self,
        ended_at: Optional[datetime] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self.weavel_client.update_trace(
            record_id=self.record_id,
            ended_at=ended_at,
        )
        self.ended_at = ended_at
        return

    def update(
        self,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        ref_record_id: Optional[str] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self.weavel_client.update_trace(
            record_id=self.record_id,
            ended_at=ended_at,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            ref_record_id=ref_record_id,
        )
        self.ended_at = ended_at
        return


class Session(BaseModel):
    """Session object."""

    user_id: Optional[str]
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
        ref_record_id: Optional[str] = None,
    ) -> Message:
        if record_id is None:
            record_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        message = Message(
            session_id=self.session_id,
            record_id=record_id,
            created_at=created_at,
            metadata=metadata,
            ref_record_id=ref_record_id,
            role=role,
            content=content,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_message(
            session_id=self.session_id,
            record_id=record_id,
            created_at=message.created_at,
            metadata=metadata,
            ref_record_id=ref_record_id,
            role=role,
            content=content,
        )
        return message

    def trace(
        self,
        name: str,
        inputs: Optional[Union[Dict[str, Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], str]] = None,
        record_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        ref_record_id: Optional[str] = None,
    ) -> Trace:
        if record_id is None:
            record_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        trace = Trace(
            session_id=self.session_id,
            record_id=record_id,
            created_at=created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            ref_record_id=ref_record_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_trace(
            session_id=self.session_id,
            record_id=record_id,
            created_at=trace.created_at,
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            ref_record_id=ref_record_id,
        )
        return trace

    def track(
        self,
        name: str,
        properties: Optional[Dict[str, str]] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        record_id: Optional[str] = None,
        ref_record_id: Optional[str] = None,
    ) -> TrackEvent:
        if record_id is None:
            record_id = str(uuid4())
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        track_event = TrackEvent(
            session_id=self.session_id,
            record_id=record_id,
            created_at=created_at,
            name=name,
            properties=properties,
            metadata=metadata,
            ref_record_id=ref_record_id,
            weavel_client=self.weavel_client,
        )
        self.weavel_client.capture_track_event(
            session_id=self.session_id,
            record_id=record_id,
            created_at=track_event.created_at,
            name=name,
            properties=properties,
            metadata=metadata,
            ref_record_id=ref_record_id,
        )
        return track_event
