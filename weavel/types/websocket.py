from enum import StrEnum
from typing import Any, Dict, Iterable, List, TypedDict, Union
from openai.types.chat.completion_create_params import ChatCompletionMessageParam


class WsLocalTask(StrEnum):
    GENERATE = "GENERATE"
    EVALUATE = "EVALUATE"
    METRIC = "METRIC"


class WsServerTask(StrEnum):
    OPTIMIZE = "OPTIMIZE"


class WsServerOptimizeResponse(StrEnum):
    OPTIMIZATION_COMPLETE = "OPTIMIZATION_COMPLETE"


class BaseWsLocalRequest(TypedDict):
    type: WsLocalTask
    correlation_id: str


class WsLocalGenerateRequest(BaseWsLocalRequest):
    prompt: Dict[str, Any]
    inputs: Dict[str, Any]


class WsLocalEvaluateRequest(BaseWsLocalRequest):
    prompt: Dict[str, Any]
    start_idx: int
    end_idx: int


class WsLocalEvaluateResponse(TypedDict):
    score: float
    results: List[Dict[str, Any]]


class WsLocalMetricRequest(BaseWsLocalRequest):
    pred: Union[str, Dict[str, Any]]
    gold: Union[str, Dict[str, Any]]
    inputs: Dict[str, Any]
    trace: Dict[str, Any]
    metadata: Dict[str, Any]
