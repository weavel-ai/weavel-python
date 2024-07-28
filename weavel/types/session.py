from typing import Dict, Optional
from datetime import datetime

from weavel._request import BaseModel


class Session(BaseModel):
    """Session object."""

    user_id: Optional[str]
    session_id: str
    created_at: datetime
    metadata: Optional[Dict[str, str]] = None
