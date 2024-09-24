from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

class DatasetItem(BaseModel):
    uuid: Optional[str] = None
    inputs: Union[Dict[str, Any], List[Any], str]
    outputs: Optional[Union[Dict[str, Any], List[Any], str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Dataset(BaseModel):
    name: str
    created_at: str
    description: Optional[str] = None
    items: List[DatasetItem]
