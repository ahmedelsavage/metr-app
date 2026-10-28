import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from stores.rag import RAG
from schemas import RetrievedChunk, ContractAnalysis
from ingest import process_file
from stores.llm import analyze_contract, summarize_contract
from pipeline import split_clauses, gather_legal_chunks, build_legal_context


def fake_generate_fn(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "json" in prompt_lower or "legal context" in prompt_lower:
        mock_response = {
            "contract_type": "other",
            "summary": "This is a mock summary generated for API testing without connecting to a real LLM server.",
            "risks": [
                {
                    "clause": "First test clause",
                    "reason": "Contains unclear terms and conditions",
                    "risk_type": "other",
                    "severity": "low",
                    "legal_reference": "Article (1) of Sample Law"
                }
            ],
            "verdict": "risky",
            "advice": "It is recommended to review the sample clauses carefully."
        }
        return json.dumps(mock_response, ensure_ascii=False)

    return "This is a mock general summary returned by Fake LLM."


class AnalyzeResponse(BaseModel):
    analysis: ContractAnalysis
    legal_context: List[RetrievedChunk]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAG()
    yield


app = FastAPI(title="Metr API", lifespan=lifespan)


def get_text_from_input(file: Optional[UploadFile], text: Optional[str]) -> str:
    if text and text.strip():
        return text.strip()

    if file:
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            extracted = process_file(tmp_path).text
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        finally:
            os.remove(tmp_path)

        if not extracted.strip():
            raise HTTPException(status_code=422, detail="No text could be extracted from the file.")
        return extracted

    raise HTTPException(status_code=400, detail="Either 'file' or 'text' must be provided.")


def run_llm(call):
    try:
        return call()
    except ValueError as error:
        raise HTTPException(status_code=502, detail=f"LLM returned an invalid response: {error}")
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=f"LLM server unavailable: {error}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=List[RetrievedChunk])
def search_rag(
    query: str = Form(..., min_length=1),
    limit: int = Form(5, ge=1, le=10),
):
    rag: RAG = app.state.rag
    return rag.search(query, limit=limit)


@app.post("/summarize")
def summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: Literal["ar", "en"] = Form("ar"),
):
    contract_text = get_text_from_input(file, text)

    summary = run_llm(
        lambda: summarize_contract(
            contract_text=contract_text,
            language=language,
            generate_fn=fake_generate_fn,
        )
    )
    return {"summary": summary}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: Literal["ar", "en"] = Form("ar"),
    limit_per_clause: int = Form(5, ge=1, le=10),
):
    rag: RAG = app.state.rag
    contract_text = get_text_from_input(file, text)

    clauses = split_clauses(contract_text)
    chunks = gather_legal_chunks(rag, clauses, limit_per_clause)
    legal_context_str = build_legal_context(chunks)

    analysis_result = run_llm(
        lambda: analyze_contract(
            contract_text=contract_text,
            legal_context=legal_context_str,
            language=language,
            generate_fn=fake_generate_fn,
        )
    )

    return AnalyzeResponse(analysis=analysis_result, legal_context=chunks)
