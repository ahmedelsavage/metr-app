from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    VECTOR_DB_PATH: str = "qdrant_storage"
    COLLECTION_NAME: str = "laws"

    EMBEDDING_BACKEND: str = "local"
    EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_URL: str = ""
    EMBEDDING_DOC_PREFIX: str = "passage: "
    EMBEDDING_QUERY_PREFIX: str = "query: "

    CHUNK_STRATEGY: str = "article"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings():
    return Settings()
