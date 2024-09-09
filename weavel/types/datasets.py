from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from ape.types import ResponseFormat

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
    

class Prompt(BaseModel):
    name: str
    description: Optional[str] = None
    created_at: str


class PromptVersion(BaseModel):
    version: int
    messages: List[Dict[str, Any]]
    model: str
    temperature: float
    response_format: Optional[ResponseFormat] = None
    input_vars: Optional[Dict[str, Any]] = None
    output_vars: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str