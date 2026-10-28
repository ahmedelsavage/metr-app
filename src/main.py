import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
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

async def get_text_from_input(file: Optional[UploadFile], text: Optional[str]) -> str:
    if text and text.strip():
        return text.strip()
        
    if file:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        try:
            ocr_result = process_file(tmp_path)
            return ocr_result.text
        finally:
            os.remove(tmp_path)
            
    raise HTTPException(status_code=400, detail="Either 'file' or 'text' must be provided.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=List[RetrievedChunk])
async def search_rag(
    query: str = Form(...),
    limit: int = Form(5)
):
    rag: RAG = app.state.rag
    return rag.search(query, limit=limit)

@app.post("/summarize")
async def summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: str = Form("ar")
):
    contract_text = await get_text_from_input(file, text)
    
    summary = summarize_contract(
        contract_text=contract_text,
        language=language,
        generate_fn=fake_generate_fn
    )
    return {"summary": summary}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    language: str = Form("ar"),
    limit_per_clause: int = Form(5)
):
    rag: RAG = app.state.rag
    contract_text = await get_text_from_input(file, text)
    
    clauses = split_clauses(contract_text)
    chunks = gather_legal_chunks(rag, clauses, limit_per_clause)
    legal_context_str = build_legal_context(chunks)
    
    analysis_result = analyze_contract(
        contract_text=contract_text,
        legal_context=legal_context_str,
        language=language,
        generate_fn=fake_generate_fn
    )
    
    return AnalyzeResponse(
        analysis=analysis_result,
        legal_context=chunks
    )
