from enum import StrEnum
from typing import Any, Dict, Iterable, List, Optional, Union
from typing_extensions import TypedDict
from openai.types.chat.completion_create_params import ChatCompletionMessageParam
from ape.common.types import DatasetItem, MetricResult, GlobalMetricResult
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


class WsLocalEvaluateResponse(TypedDict):
    score: Optional[float]
    preds: Optional[List[Union[str, Dict[str, Any]]]]
    eval_results: Optional[List[MetricResult]]
    global_result: Optional[GlobalMetricResult]


class WsLocalMetricRequest(BaseWsLocalRequest):
    dataset_item: DatasetItem
    pred: Union[str, Dict[str, Any]]
