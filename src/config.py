from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


SRC_DIR = Path(__file__).resolve().parent
ENV_FILE = SRC_DIR / ".env"


class Settings(BaseSettings):
    # RAG settings
    VECTOR_DB_PATH: str = "qdrant_storage"
    COLLECTION_NAME: str = "laws"
    EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-small"

    # LLM settings
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = ""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
    )


def get_settings():
    return Settings()
