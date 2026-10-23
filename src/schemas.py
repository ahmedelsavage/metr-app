from typing import Optional
from pydantic import BaseModel

class RetrievedChunk(BaseModel):
    text: str
    score: float
    law_name: str
    article_number: Optional[str] = None
