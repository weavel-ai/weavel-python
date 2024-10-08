from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union
from typing_extensions import TypedDict
from openai.types.chat.completion_create_params import ChatCompletionMessageParam
from ape.common.types import DatasetItem, MetricResult, GlobalMetricResult

class WsLocalTask(str, Enum):
    GENERATE = "GENERATE"
    EVALUATE = "EVALUATE"
    METRIC = "METRIC"
    GLOBAL_METRIC = "GLOBAL_METRIC"


class WsServerTask(str, Enum):
    OPTIMIZE = "OPTIMIZE"

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

class WsLocalGlobalMetricRequest(BaseWsLocalRequest):
    results: List[MetricResult]
    
class WsLocalGlobalMetricResponse(TypedDict):
    global_result: GlobalMetricResult

class WsLocalMetricRequest(BaseWsLocalRequest):
    dataset_item: DatasetItem
    pred: Union[str, Dict[str, Any]]
