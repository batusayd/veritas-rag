from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings validated via Pydantic v2 BaseSettings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Runtime
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS",
    )

    # LLM & Dense Embedding Providers
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    query_rewriter_model: str = Field(default="gpt-4o-mini", alias="QUERY_REWRITER_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")
    qdrant_local_path: Optional[str] = Field(
        default="./qdrant_local_storage",
        alias="QDRANT_LOCAL_PATH",
    )
    qdrant_collection_name: str = Field(
        default="document_intelligence_hybrid",
        alias="QDRANT_COLLECTION_NAME",
    )
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # Hybrid Search, Chunking & Reranking
    sparse_model_name: str = Field(default="Qdrant/bm25", alias="SPARSE_MODEL_NAME")
    reranker_model_name: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        alias="RERANKER_MODEL_NAME",
    )
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    hybrid_prefetch_limit: int = Field(default=20, alias="HYBRID_PREFETCH_LIMIT")
    rrf_top_k: int = Field(default=12, alias="RRF_TOP_K")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    rerank_score_threshold: float = Field(default=0.20, alias="RERANK_SCORE_THRESHOLD")


@lru_cache
def get_settings() -> Settings:
    return Settings()
