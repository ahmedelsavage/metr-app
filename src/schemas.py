from typing import List, Literal, Optional
from pydantic import BaseModel, model_validator


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


RiskType = Literal[
    "unilateral_change",
    "unilateral_termination",
    "unlimited_liability",
    "liability_exclusion",
    "unfair_arbitration",
    "unfair_deduction",
    "unfair_content_removal",
    "other",
]


# LLM output-1
class RiskItem(BaseModel):
    clause: str
    reason: str
    risk_type: RiskType
    severity: Literal["low", "medium", "high"]
    legal_reference: Optional[str] = None


# LLM Output-2 (final output)
class ContractAnalysis(BaseModel):
    contract_type: Literal["sale", "rent", "other"]
    summary: str
    risks: List[RiskItem]
    # "safe" = no clear risks found. "risky" = at least one risk.
    verdict: Literal["safe", "risky"]
    advice: str

    @model_validator(mode="after")
    def verdict_must_match_risks(self):
        if self.risks and self.verdict == "safe":
            raise ValueError(
                "verdict is 'safe' but the risks list is not empty. "
                "Use 'risky' when there is at least one risk."
            )

        if not self.risks and self.verdict == "risky":
            raise ValueError(
                "verdict is 'risky' but the risks list is empty. "
                "Use 'safe' when there are no risks."
            )

        return self


# BERT Output
class BERTClauseResult(BaseModel):
    clause: str
    is_unfair: bool
    labels: List[str]