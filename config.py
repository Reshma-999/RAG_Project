"""
config.py
─────────
Central configuration loaded from environment variables via pydantic-settings.
All tunable knobs live here so nothing is hard-coded elsewhere.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_password: str = Field("", alias="REDIS_PASSWORD")
    redis_db: int = Field(0, alias="REDIS_DB")
    cache_similarity_threshold: float = Field(0.92, alias="CACHE_SIMILARITY_THRESHOLD")
    cache_ttl: int = Field(86400, alias="CACHE_TTL")

    # ── Vector Store ──────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        Path("./data/chroma_db"), alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field("rag_documents", alias="CHROMA_COLLECTION_NAME")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    data_dir: Path = Field(Path("./data/raw"), alias="DATA_DIR")
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    batch_size: int = Field(500, alias="BATCH_SIZE")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(5, alias="TOP_K")
    retrieval_score_threshold: float = Field(0.35, alias="RETRIEVAL_SCORE_THRESHOLD")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_dataset_path: Path = Field(
        Path("./data/eval_dataset.json"), alias="EVAL_DATASET_PATH"
    )
    eval_results_dir: Path = Field(
        Path("./data/eval_results"), alias="EVAL_RESULTS_DIR"
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
