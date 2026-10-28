from contextlib import asynccontextmanager
from fastapi import FastAPI
from stores.rag import RAG

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAG()
    yield

app = FastAPI(title="Metr API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

