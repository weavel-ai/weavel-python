from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel
from ape.common.types import ResponseFormat


class WvPrompt(BaseModel):
    uuid: Optional[str] = None
    name: str
    description: Optional[str] = None
    created_at: str


class WvPromptVersion(BaseModel):
    uuid: Optional[str] = None
    prompt_uuid: Optional[str] = None
    version: int
    messages: List[Dict[str, Any]]
    model: str
    temperature: float
    response_format: Optional[ResponseFormat] = None
    input_vars: Optional[Dict[str, Any]] = None
    output_vars: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
