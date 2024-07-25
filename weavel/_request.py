from typing import List, Union, Literal
from ._body import (
    BaseModel,
    IngestionType,
    IngestionBody,
    CaptureSessionBody,
    CaptureMessageBody,
    CaptureTrackEventBody,
    CaptureTraceBody,
    CaptureSpanBody,
    CaptureLogBody,
    CaptureGenerationBody,
    UpdateTraceBody,
    UpdateSpanBody,
    UpdateGenerationBody,
    IdentifyUserBody,
    CaptureTestObservationBody,
)


class CaptureSessionRequest(IngestionBody):
    """
    Represents a request to capture a session.

    Attributes:
        type (Literal[IngestionType.OpenSession]): The type of ingestion, set to IngestionType.OpenSession.
        body (OpenSessionBody): The body of the open session request.
    """

    type: Literal[IngestionType.CaptureSession] = IngestionType.CaptureSession
    body: CaptureSessionBody


class CaptureMessageRequest(IngestionBody):
    """
    Represents a request to capture a message.

    Attributes:
        type (Literal[IngestionType.CaptureMessage]): The type of ingestion, which is set to IngestionType.CaptureMessage.
        body (CaptureMessageBody): The body of the capture message request.
    """

    type: Literal[IngestionType.CaptureMessage] = IngestionType.CaptureMessage
    body: CaptureMessageBody


class CaptureTrackEventRequest(IngestionBody):
    """
    Represents a request to capture a track event.

    Attributes:
        type (Literal[IngestionType.CaptureTrackEvent]): The type of ingestion, set to IngestionType.CaptureTrackEvent.
        body (CaptureTrackEventBody): The body of the capture track event request.
    """

    type: Literal[IngestionType.CaptureTrackEvent] = IngestionType.CaptureTrackEvent
    body: CaptureTrackEventBody


class CaptureTraceRequest(IngestionBody):
    """
    Represents a request to capture a trace.

    Attributes:
        type (Literal[IngestionType.CaptureTrace]): The type of ingestion, set to IngestionType.CaptureTrace.
        body (CaptureTraceBody): The body of the capture trace request.
    """

    type: Literal[IngestionType.CaptureTrace] = IngestionType.CaptureTrace
    body: CaptureTraceBody


class CaptureSpanRequest(IngestionBody):
    """
    Represents a request to capture a span.

    Attributes:
        type (Literal[IngestionType.CaptureSpan]): The type of ingestion, set to IngestionType.CaptureSpan.
        body (CaptureSpanBody): The body of the capture span request.
    """

    type: Literal[IngestionType.CaptureSpan] = IngestionType.CaptureSpan
    body: CaptureSpanBody


class CaptureLogRequest(IngestionBody):
    """
    Represents a request to capture log data.

    Attributes:
        type (Literal[IngestionType.CaptureLog]): The type of ingestion, set to IngestionType.CaptureLog.
        body (CaptureLogBody): The body of the capture log request.
    """

    type: Literal[IngestionType.CaptureLog] = IngestionType.CaptureLog
    body: CaptureLogBody


class CaptureGenerationRequest(IngestionBody):
    """
    Represents a request for capture generation.

    Attributes:
        type (Literal[IngestionType.CaptureGeneration]): The type of ingestion, which is set to IngestionType.CaptureGeneration.
        body (CaptureGenerationBody): The body of the capture generation request.
    """

    type: Literal[IngestionType.CaptureGeneration] = IngestionType.CaptureGeneration
    body: CaptureGenerationBody

class CaptureTestObservationRequest(IngestionBody):
    type: Literal[IngestionType.CaptureTestObservation] = IngestionType.CaptureTestObservation
    body: CaptureTestObservationBody

class UpdateTraceRequest(IngestionBody):
    type: Literal[IngestionType.UpdateTrace] = IngestionType.UpdateTrace
    body: UpdateTraceBody


class UpdateSpanRequest(IngestionBody):
    type: Literal[IngestionType.UpdateSpan] = IngestionType.UpdateSpan
    body: UpdateSpanBody



class IdentifyUserRequest(IngestionBody):
    """
    Represents a request to identify a user.

    Attributes:
        type (Literal[IngestionType.IdentifyUser]): The type of ingestion, which is set to IngestionType.IdentifyUser.
        body (IdentifyUserBody): The body of the identification request.
    """

    type: Literal[IngestionType.IdentifyUser] = IngestionType.IdentifyUser
    body: IdentifyUserBody


class UpdateGenerationRequest(IngestionBody):
    type: Literal[IngestionType.UpdateGeneration] = IngestionType.UpdateGeneration
    body: UpdateGenerationBody


class BatchRequest(BaseModel):
    batch: List[
        Union[
            CaptureSessionRequest,
            IdentifyUserRequest,
            CaptureMessageRequest,
            CaptureTrackEventRequest,
            CaptureTraceRequest,
            CaptureSpanRequest,
            CaptureLogRequest,
            CaptureGenerationRequest,
            UpdateTraceRequest,
            UpdateSpanRequest,
            UpdateGenerationRequest,
        ]
    ]
    
UnionRequest = Union[
    CaptureSessionRequest,
    IdentifyUserRequest,
    CaptureMessageRequest,
    CaptureTrackEventRequest,
    CaptureTraceRequest,
    CaptureSpanRequest,
    CaptureLogRequest,
    CaptureGenerationRequest,
    CaptureTestObservationRequest,
    UpdateTraceRequest,
    UpdateSpanRequest,
    UpdateGenerationRequest,
]