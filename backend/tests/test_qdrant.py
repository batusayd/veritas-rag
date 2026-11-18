from unittest.mock import AsyncMock
import pytest
from qdrant_client import models

from app.services.parser import DocumentParser
from app.services.qdrant_service import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantHybridService,
)


@pytest.mark.asyncio
async def test_ensure_collection_creates_dense_and_sparse_config(
    qdrant_service: QdrantHybridService,
    mock_qdrant_client: AsyncMock,
):
    await qdrant_service.ensure_collection()

    mock_qdrant_client.create_collection.assert_awaited_once()
    call_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
    assert DENSE_VECTOR_NAME in call_kwargs["vectors_config"]
    assert SPARSE_VECTOR_NAME in call_kwargs["sparse_vectors_config"]
    assert mock_qdrant_client.create_payload_index.await_count == 2


@pytest.mark.asyncio
async def test_check_file_hash_idempotency(
    qdrant_service: QdrantHybridService,
    mock_qdrant_client: AsyncMock,
):
    # Initially 0 chunks exist
    exists, count = await qdrant_service.check_file_hash_exists("deadbeef123")
    assert exists is False
    assert count == 0

    # Simulate already-ingested document with 8 chunks
    mock_qdrant_client.count = AsyncMock(return_value=models.CountResult(count=8))
    exists_after, count_after = await qdrant_service.check_file_hash_exists("deadbeef123")
    assert exists_after is True
    assert count_after == 8


@pytest.mark.asyncio
async def test_index_chunks_upserts_hybrid_points(
    parser: DocumentParser,
    qdrant_service: QdrantHybridService,
    mock_qdrant_client: AsyncMock,
):
    _, _, chunks = parser.parse_document(
        file_bytes=b"# Section One\nHybrid RAG combines dense and sparse representations.",
        filename="notes.md",
    )

    indexed = await qdrant_service.index_chunks(chunks)
    assert indexed == len(chunks)
    mock_qdrant_client.upsert.assert_awaited_once()

    upserted_points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    first_point = upserted_points[0]
    assert DENSE_VECTOR_NAME in first_point.vector
    assert SPARSE_VECTOR_NAME in first_point.vector
    assert first_point.payload["source_file"] == "notes.md"


@pytest.mark.asyncio
async def test_hybrid_search_executes_rrf_fusion_query(
    qdrant_service: QdrantHybridService,
    mock_qdrant_client: AsyncMock,
):
    candidates, embed_ms, search_ms = await qdrant_service.hybrid_search(
        query_text="How does Reciprocal Rank Fusion work?",
        source_filter="architecture.md",
    )

    assert len(candidates) == 1
    assert candidates[0]["source_file"] == "architecture.md"
    assert candidates[0]["rrf_score"] == pytest.approx(0.8421)
    assert embed_ms >= 0.0
    assert search_ms >= 0.0

    query_kwargs = mock_qdrant_client.query_points.call_args.kwargs
    assert len(query_kwargs["prefetch"]) == 2
    assert query_kwargs["query"].fusion == models.Fusion.RRF
