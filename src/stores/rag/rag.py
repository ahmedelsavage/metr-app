from pathlib import Path
from typing import List, Optional

import pymupdf
from qdrant_client import QdrantClient, models

from config import get_settings
from schemas import RetrievedChunk
from .chunking import chunk_text
from .embeddings import Embedder


class RAG:
    def __init__(self):
        self.s = get_settings()
        self.client = QdrantClient(path=self.s.VECTOR_DB_PATH)
        self.embedder = Embedder(self.s)

    def collection(self, strategy: Optional[str] = None) -> str:
        model = self.s.EMBEDDING_MODEL_ID.split("/")[-1].replace(".", "_").replace("-", "_")
        return f"{self.s.COLLECTION_NAME}_{strategy or self.s.CHUNK_STRATEGY}_{model}"

    def read_file(self, path: Path) -> Optional[str]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "\n".join(page.get_text() for page in pymupdf.open(path))
        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        return None

    def build_kb(self, folder: str, strategy: Optional[str] = None) -> int:
        strategy = strategy or self.s.CHUNK_STRATEGY
        name = self.collection(strategy)

        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=self.embedder.size, distance=models.Distance.COSINE),
        )

        chunks = []
        for f in sorted(Path(folder).iterdir()):
            text = self.read_file(f)
            if text is None:
                continue
            for c in chunk_text(text, strategy, self.s.CHUNK_SIZE, self.s.CHUNK_OVERLAP):
                if not c.get("law_name"):
                    c["law_name"] = f.stem
                chunks.append(c)

        if not chunks:
            return 0

        vectors = self.embedder.embed_documents([c["text"] for c in chunks])
        points = [models.PointStruct(id=i, vector=vectors[i], payload=chunks[i]) for i in range(len(chunks))]
        for i in range(0, len(points), 64):
            self.client.upsert(collection_name=name, points=points[i:i + 64])

        return len(chunks)

    def search(self, query: str, limit: int = 5, strategy: Optional[str] = None) -> List[RetrievedChunk]:
        name = self.collection(strategy)
        if not self.client.collection_exists(name):
            return []

        result = self.client.query_points(
            collection_name=name,
            query=self.embedder.embed_query(query),
            limit=limit,
        )
        return [
            RetrievedChunk(
                text=p.payload["text"],
                score=p.score,
                law_name=p.payload["law_name"],
                article_number=p.payload.get("article_number"),
            )
            for p in result.points
        ]
