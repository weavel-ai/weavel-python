from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DatasetItems(BaseModel):
    inputs: Dict[str, str]
    outputs: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetDetails(BaseModel):
    dataset_details: Dict[str, Any]
    dataset_items: List[DatasetItems]

