from typing import List, Literal, Optional
from pydantic import BaseModel

# OCR Output
class OCRResult(BaseModel):
    text: str
    confidence: float = 1.0
    language: str = "ar"

# RAG Output

class RetrievedChunk(BaseModel):
    text: str
    score: float
    law_name: str
    article_number: Optional[str] = None

# LLM output-1
class RiskItem(BaseModel):
    clause: str
    reason: str
    severity: Literal["low", "medium", "high"]
    legal_reference: Optional[str] = None

# LLM Output-2 (final output)
class ContractAnalysis(BaseModel):
    contract_type: Literal["sale", "rent", "other"]
    summary: str
    risks: List[RiskItem]
    advice: str
