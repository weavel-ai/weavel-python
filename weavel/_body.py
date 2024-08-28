from h11 import Data
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Any, Dict, List, Optional, Literal, Union
from enum import Enum


class BaseModel(PydanticBaseModel):
    """Extended Pydantic BaseModel"""

    class Config:
        arbitrary_types_allowed = True


class IngestionType(str, Enum):
    CaptureSession = "capture-session"
    IdentifyUser = "identify-user"
    CaptureMessage = "capture-message"
    CaptureTrackEvent = "capture-track-event"
    CaptureTrace = "capture-trace"
    CaptureSpan = "capture-span"
    CaptureLog = "capture-log"
    CaptureGeneration = "capture-generation"
    CaptureTestObservation = "capture-test-observation"
    UpdateTrace = "update-trace"
    UpdateSpan = "update-span"
    UpdateGeneration = "update-generation"


class IngestionBody(BaseModel):
    type: IngestionType


class BaseRecordBody(BaseModel):
    """
    Represents the base record body.

    Attributes:
        session_id (str, optional): The unique identifier for the session. Optional.
        record_id (str, optional): The unique identifier for the record. Optional.
        created_at (str, optional): The datetime when the record was captured. Optional.
        metadata (Optional[Dict[str, Any]]): Additional metadata associated with the record. Optional.
        ref_record_id (str, optional): The record ID to reference. Optional.
    """

    session_id: Optional[str] = Field(
        default=None, description="The unique identifier for the record."
    )
    record_id: Optional[str] = Field(
        default=None,
        description="The unique identifier for the record. Optional.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="The datetime when the record was captured. Optional.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata associated with the record. Optional.",
    )
    ref_record_id: Optional[str] = Field(
        default=None,
        description="The record ID to reference. Optional.",
    )


class BaseObservationBody(BaseModel):
    """
    Represents the base observation body.

    Attributes:
        record_id (str): The unique identifier for the record.
        observation_id (str, optional): The unique identifier for the observation. Optional.
        created_at (str, optional): The datetime when the observation was captured. Optional.
        name (str): The name of the observation.
        parent_observation_id (str, optional): The parent observation ID. Optional.
    """

    record_id: Optional[str] = Field(
        default=None, description="The unique identifier for the record."
    )
    observation_id: Optional[str] = Field(
        default=None,
        description="The unique identifier for the observation. Optional.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="The datetime when the observation was captured. Optional.",
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the observation. Optional.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata associated with the observation. Optional.",
    )
    parent_observation_id: Optional[str] = Field(
        default=None,
        description="The parent observation ID. Optional.",
    )


class CaptureSessionBody(BaseModel):
    """
    Represents the request body for capturing a session.

    Attributes:
        user_id (str, optional): The unique identifier for the user.
        session_id (str, optional): The unique identifier for the session data.
        metadata (Dict[str, Any], Optional): Additional metadata associated with the session. Optional.
        created_at (str, optional): The datetime when the session was opened. Optional.
    """

    user_id: Optional[str] = Field(
        default=None, description="The unique identifier for the user."
    )
    session_id: Optional[str] = Field(
        default=None, description="The unique identifier for the session data."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional metadata associated with the session. Optional.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="The datetime when the session was opened. Optional.",
    )


class CaptureRecordBody(BaseRecordBody):
    session_id: str


class CaptureObservationBody(BaseObservationBody):
    name: Optional[str] = Field(
        default=None,
        description="The name of the observation. Optional.",
    )


class CaptureMessageBody(CaptureRecordBody):
    """
    Represents a capture message body.

    Attributes:
        role (Literal["user", "assistant", "system"]): The role of the session, can be 'user', 'assistant', or 'system'.
        content (str): The content of the record.
    """

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="The role of the session, 'user', 'assistant', 'system'.",
    )
    content: str = Field(..., description="The content of the record.")


class CaptureTrackEventBody(CaptureRecordBody):
    """
    Represents the body of a capture track event.

    Attributes:
        name (str): The name of the record. Optional.
        properties (Optional[Dict[str, Any]]): Additional properties associated with the record. Optional.
    """

    name: str = Field(
        ...,
        description="The name of the record. Optional.",
    )
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional properties associated with the record. Optional.",
    )


class CaptureTraceBody(CaptureRecordBody):
    """
    Represents the body of a capture trace record.

    Attributes:
        name (str): The name of the record. Optional.
    """

    name: str = Field(
        ...,
        description="The name of the trace record. Optional.",
    )
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The inputs of the trace. Optional.",
    )
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The outputs of the trace. Optional.",
    )


class CaptureSpanBody(CaptureObservationBody):
    """
    Represents the body of a capture span observation.

    Attributes:
        inputs (Optional[Union[Dict[str, Any], List[Any], str]]): The inputs of the generation. Optional.
        outputs (Optional[Union[Dict[str, Any], List[Any], str]]): The outputs of the generation. Optional.
    """

    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The inputs of the generation. Optional.",
    )
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The outputs of the generation. Optional.",
    )


class CaptureTestObservationBody(CaptureObservationBody):
    test_uuid: str = Field(
        ...,
        description="The unique identifier for the test.",
    )
    dataset_item_uuid: str = Field(
        ...,
        description="The unique identifier for the dataset item.",
    )
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The inputs of the generation. Optional.",
    )
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The outputs of the generation. Optional.",
    )


class CaptureLogBody(CaptureObservationBody):
    """
    Represents the body of a capture log observation.

    Attributes:
        value (str, optional): The value of the observation. Optional.
    """

    value: Optional[str] = Field(
        default=None,
        description="The value of the observation. Optional.",
    )


class CaptureGenerationBody(CaptureObservationBody):
    """
    Represents the body of a capture generation.

    Attributes:
        inputs (Optional[Union[Dict[str, Any], List[Any], str]]): The inputs of the generation. Optional.
        outputs (Optional[Union[Dict[str, Any], List[Any], str]]): The outputs of the generation. Optional.
        prompt_name (Optional[str]): The name of the prompt. Optional.
    """

    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The inputs of the generation. Optional.",
    )
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        default=None,
        description="The outputs of the generation. Optional.",
    )
    messages: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="The messages of the generation. Optional.",
    )
    model: Optional[str] = Field(
        default=None,
        description="The model of the generation. Optional.",
    )
    latency: Optional[float] = Field(
        default=None,
        description="The latency of the generation. Optional.",
    )
    cost: Optional[float] = Field(
        default=None,
        description="The cost of the generation. Optional.",
    )
    prompt_name: Optional[str] = Field(
        default=None,
        description="The name of the prompt. Optional.",
    )


class UpdateTraceBody(BaseRecordBody):
    record_id: str
    ended_at: Optional[str] = None
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateSpanBody(BaseObservationBody):
    observation_id: str
    ended_at: Optional[str] = None
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateGenerationBody(BaseObservationBody):
    observation_id: str
    ended_at: Optional[str] = None
    inputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    metadata: Optional[Dict[str, Any]] = None


class IdentifyUserBody(BaseModel):
    """
    Represents the body of a request to identify a user.

    Attributes:
        user_id (str): The unique identifier for the user.
        properties (Dict[str, Any], optional): Additional properties associated with the track event. Optional.
        created_at (Optional[str], optional): The datetime when the User is identified. Optional.
    """

    user_id: str = Field(..., description="The unique identifier for the user.")
    properties: Dict[str, Any] = Field(
        description="Additional properties associated with the track event. Optional.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="The datetime when the User is identified. Optional.",
    )
