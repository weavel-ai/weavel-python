from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from enum import Enum
from uuid import UUID
from datetime import datetime, date

class WeavelObject(BaseModel):
    """Weavel Object. Extended Pydantic BaseModel with support for UUID and datetime."""
    def __init__(self, **data: Any):
        for key, value in data.items():
            if key in self.__annotations__ and (self.__annotations__[key] == str or self.__annotations__[key] == Optional[str]):
                if (
                    isinstance(value, UUID)
                    or isinstance(value, datetime)
                    or isinstance(value, date)
                ):
                    try:
                        data[key] = str(value)
                    except:
                        data[key] = value
        super().__init__(**data)
        
        
class DataType(str, Enum):
    user_message = "user_message"
    assistant_message = "assistant_message"
    inner_step = "inner_step"
    retrieved_content = "retrieved_content"

class BackgroundTaskType(str, Enum):
    start_trace = "start_trace"
    log_trace_data = "log_trace_data"
    save_metadata_trace = "save_metadata_trace"
    save_metadata_trace_data = "save_metadata_trace_data"
    create_tags_trace_data = "create_tags_trace_data"
    
class StartTraceBody(WeavelObject):
    """Start Trace body."""
    user_uuid: str
    trace_uuid: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

class SaveTraceDataBody(WeavelObject):
    """Log body."""
    trace_uuid: str
    data_type: str
    data_content: str
    timestamp: Optional[str] = None
    unit_name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    
class SaveMetadataTraceBody(WeavelObject):
    """Save metadata body."""
    trace_uuid: str
    metadata: Dict[str, str]

class SaveMetadataTraceDataBody(WeavelObject):
    """Save trace_data metadata body."""
    trace_uuid: str
    data_type: str
    data_content: str
    metadata: Dict[str, str]
    
class WeavelRequest(WeavelObject):
    """Weavel Request."""
    task: str
    body: Union[StartTraceBody, SaveTraceDataBody, SaveMetadataTraceBody, SaveMetadataTraceDataBody]
    
    @validator('task', pre=True)
    def validate_task(cls, v):
        if v not in BackgroundTaskType.__members__:
            raise ValueError(f"Invalid task type: {v}. Must be one of {list(BackgroundTaskType.__members__.keys())}")
        return v
    
class BatchRequest(WeavelObject):
    requests: list[WeavelRequest]
