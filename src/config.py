from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    VECTOR_DB_PATH: str = "qdrant_storage"
    COLLECTION_NAME: str = "laws"
    EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-small"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings():
    return Settings()
