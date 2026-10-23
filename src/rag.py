import re
from pathlib import Path
from typing import List

import pymupdf
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from config import get_settings
from schemas import RetrievedChunk

ARTICLE_SPLIT = re.compile(r"(?=^\s*(?:\(?\s*ماد[ةه]\s*\(?\s*[0-9٠-٩۰-۹]+|[0-9٠-٩۰-۹]+\s*[-–.)]))", re.MULTILINE)
ARTICLE_NUMBER = re.compile(r"(?:ماد[ةه]\s*\(?\s*)?([0-9٠-٩۰-۹]+)")


class RAG:
    def __init__(self):
        settings = get_settings()
        self.collection = settings.COLLECTION_NAME
        self.client = QdrantClient(path=settings.VECTOR_DB_PATH)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
        self.size = self.model.get_embedding_dimension()
        
    def read_pdf(self, path: Path) -> str:
        doc = pymupdf.open(path)
        return "\n".join(page.get_text() for page in doc)

    def chunk_by_article(self, text: str, law_name: str) -> List[dict]:
        chunks = []
        for part in ARTICLE_SPLIT.split(text):
            part = part.strip()
            if len(part) < 15:
                continue
            match = ARTICLE_NUMBER.search(part)
            chunks.append({
                "text": part,
                "law_name": law_name,
                "article_number": match.group(1) if match else None,
            })
        return chunks

    def embed(self, texts: List[str], prefix: str) -> List[List[float]]:
        return self.model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            batch_size=16,
        ).tolist()

    def build_kb(self, folder: str, reset: bool = True) -> int:
        if reset and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.size,
                    distance=models.Distance.COSINE,
                ),
            )

        chunks = []
        for f in sorted(Path(folder).iterdir()):
            if f.suffix.lower() == ".pdf":
                text = self.read_pdf(f)
            elif f.suffix.lower() == ".txt":
                text = f.read_text(encoding="utf-8")
            else:
                continue
            chunks.extend(self.chunk_by_article(text, law_name=f.stem))

        if not chunks:
            return 0

        vectors = self.embed([c["text"] for c in chunks], "passage: ")
        points = [
            models.PointStruct(id=i, vector=vectors[i], payload=chunks[i])
            for i in range(len(chunks))
        ]

        for i in range(0, len(points), 64):
            self.client.upsert(collection_name=self.collection, points=points[i:i + 64])

        return len(chunks)

    def search(self, query: str, limit: int = 5) -> List[RetrievedChunk]:
        if not self.client.collection_exists(self.collection):
            return []

        vector = self.embed([query], "query: ")[0]
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
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
