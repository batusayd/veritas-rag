from unittest.mock import AsyncMock, MagicMock
import pytest
from qdrant_client import models

from app.config import Settings
from app.services.parser import DocumentParser
from app.services.qdrant_service import QdrantHybridService


@pytest.fixture
def test_settings() -> Settings:
    """Isolated settings fixture for unit testing without external API dependencies."""
    return Settings(
        APP_ENV="test",
        OPENAI_API_KEY="sk-your-placeholder-key",
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        QDRANT_COLLECTION_NAME="test_hybrid_collection",
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=40,
        RERANK_TOP_K=3,
        RERANK_SCORE_THRESHOLD=0.20,
    )


@pytest.fixture
def parser(test_settings: Settings) -> DocumentParser:
    return DocumentParser(
        chunk_size=test_settings.chunk_size,
        chunk_overlap=test_settings.chunk_overlap,
    )


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock(return_value=True)
    client.create_payload_index = AsyncMock(return_value=True)
    client.count = AsyncMock(return_value=models.CountResult(count=0))
    client.upsert = AsyncMock(return_value=True)

    scored_point = MagicMock()
    scored_point.id = "11111111-2222-3333-4444-555555555555"
    scored_point.score = 0.8421
    scored_point.payload = {
        "text": "Reciprocal Rank Fusion (RRF) combines dense cosine vectors and sparse BM25 scores.",
        "source_file": "architecture.md",
        "file_hash": "abc123hash",
        "page_number": 1,
        "section_title": "Hybrid Search Engine",
        "chunk_index": 0,
    }
    query_res = MagicMock()
    query_res.points = [scored_point]
    client.query_points = AsyncMock(return_value=query_res)
    return client


@pytest.fixture
def qdrant_service(
    test_settings: Settings,
    mock_qdrant_client: AsyncMock,
) -> QdrantHybridService:
    service = QdrantHybridService(settings=test_settings, client=mock_qdrant_client)
    # Stub out sparse model download for fast unit tests
    service.generate_sparse_embeddings = AsyncMock(
        side_effect=lambda texts: [
            models.SparseVector(indices=[1, 42, 108], values=[0.85, 1.42, 0.63])
            for _ in texts
        ]
    )
    return service
