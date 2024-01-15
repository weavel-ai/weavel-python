from pydantic import BaseModel
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
    