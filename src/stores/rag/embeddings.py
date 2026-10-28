from typing import List
from config import Settings


class Embedder:
    def __init__(self, s: Settings):
        self.backend = s.EMBEDDING_BACKEND
        self.model_id = s.EMBEDDING_MODEL_ID
        self.doc_prefix = s.EMBEDDING_DOC_PREFIX
        self.query_prefix = s.EMBEDDING_QUERY_PREFIX

        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_id)
        elif self.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=s.EMBEDDING_API_KEY, base_url=s.EMBEDDING_API_URL or None)
        elif self.backend == "cohere":
            import cohere
            self.client = cohere.ClientV2(api_key=s.EMBEDDING_API_KEY)
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {self.backend}")

        self.size = len(self.embed_query("test"))

    def _encode(self, texts: List[str], is_query: bool) -> List[List[float]]:
        prefix = self.query_prefix if is_query else self.doc_prefix
        texts = [prefix + t for t in texts]

        if self.backend == "local":
            return self.model.encode(texts, normalize_embeddings=True, batch_size=16).tolist()

        out = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            if self.backend == "openai":
                r = self.client.embeddings.create(model=self.model_id, input=batch)
                out += [d.embedding for d in r.data]
            else:
                r = self.client.embed(
                    texts=batch,
                    model=self.model_id,
                    input_type="search_query" if is_query else "search_document",
                    embedding_types=["float"],
                )
                out += r.embeddings.float_
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], is_query=True)[0]
