from typing import Dict, List, Optional, Any, Literal, Union
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field, PrivateAttr

from typing import Literal
from weavel._worker import Worker
from .types import Session, Message, Trace, TrackEvent, Log, Span, Generation


class BaseClient(BaseModel):
    _weavel_client: Worker = PrivateAttr()

    def __init__(self, weavel_client: Worker, **data):
        super().__init__(**data)
        self._weavel_client = weavel_client


class ObjectClient(BaseClient):
    record_id: Optional[str] = None

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
        )
        self._weavel_client.capture_log(
            **{
                k: v
                for k, v in log.model_dump().items()
                if k != "type" and v is not None
            }
        )
        return log

    def generation(
        self,
        name: str,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "GenerationClient":
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
            parent_observation_id=(
                self.observation_id if isinstance(self, SpanClient) else None
            ),
        )
        generation_client = GenerationClient(
            **generation.model_dump(),
            weavel_client=self._weavel_client,
        )
        self._weavel_client.capture_generation(
            **{
                k: v
                for k, v in generation.model_dump().items()
                if k != "type" and v is not None
            }
        )

        return generation_client

    def span(
        self,
        name: str,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        observation_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "SpanClient":
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
            parent_observation_id=(
                self.observation_id if isinstance(self, SpanClient) else None
            ),
        )
        span_client = SpanClient(
            **span.model_dump(),
            weavel_client=self._weavel_client,
        )
        self._weavel_client.capture_span(
            **{
                k: v
                for k, v in span.model_dump().items()
                if k != "type" and v is not None
            }
        )

        return span_client


class SessionClient(Session, BaseClient):
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
        )
        self._weavel_client.capture_message(
            **{
                k: v
                for k, v in message.model_dump().items()
                if k != "type" and v is not None
            }
        )
        return message

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
        )
        self._weavel_client.capture_track_event(
            **{
                k: v
                for k, v in track_event.model_dump().items()
                if k != "type" and v is not None
            }
        )
        return track_event

    def trace(
        self,
        name: str,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        record_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        ref_record_id: Optional[str] = None,
    ) -> "TraceClient":
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
        )
        trace_client = TraceClient(
            **trace.model_dump(),
            weavel_client=self._weavel_client,
        )
        self._weavel_client.capture_trace(
            **{
                k: v
                for k, v in trace.model_dump().items()
                if k != "type" and v is not None
            }
        )
        return trace_client


class TraceClient(Trace, ObjectClient):

    def update(
        self,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        ref_record_id: Optional[str] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self._weavel_client.update_trace(
            record_id=self.record_id,
            ended_at=ended_at,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            ref_record_id=ref_record_id,
        )
        self.ended_at = ended_at
        return

    def end(
        self,
        ended_at: Optional[datetime] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self._weavel_client.update_trace(
            record_id=self.record_id,
            ended_at=ended_at,
        )
        self.ended_at = ended_at
        return


class SpanClient(Span, ObjectClient):

    def update(
        self,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self._weavel_client.update_span(
            observation_id=self.observation_id,
            ended_at=ended_at,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        self.ended_at = ended_at
        return

    def end(
        self,
        ended_at: Optional[datetime] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self._weavel_client.update_span(
            observation_id=self.observation_id,
            ended_at=ended_at,
        )
        self.ended_at = ended_at
        return


class GenerationClient(Generation, ObjectClient):

    def update(
        self,
        ended_at: Optional[datetime] = None,
        inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self._weavel_client.update_generation(
            observation_id=self.observation_id,
            ended_at=ended_at,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        self.ended_at = ended_at
        return

    def end(
        self,
        ended_at: Optional[datetime] = None,
    ) -> None:
        if ended_at is None:
            ended_at = datetime.now(timezone.utc)

        self._weavel_client.update_generation(
            observation_id=self.observation_id,
            ended_at=ended_at,
        )
        self.ended_at = ended_at
        return
