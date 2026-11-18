import asyncio
import hashlib
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from fastembed import SparseTextEmbedding
import litellm
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings, get_settings
from app.models.schemas import DocumentChunk, DocumentSummary

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantHybridService:
    """
    Production Qdrant service managing dual-vector (Dense + Sparse BM25) indexing,
    SHA-256 file hash idempotency checks, and Reciprocal Rank Fusion (RRF) Hybrid Search.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[AsyncQdrantClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = self.settings.qdrant_collection_name
        if client is not None:
            self.client = client
        elif self.settings.qdrant_local_path:
            logger.info("Using embedded local Qdrant storage at: %s", self.settings.qdrant_local_path)
            self.client = AsyncQdrantClient(path=self.settings.qdrant_local_path)
        else:
            self.client = AsyncQdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                api_key=self.settings.qdrant_api_key or None,
                timeout=30.0,
            )
        self._sparse_model: Optional[SparseTextEmbedding] = None
        self._collection_initialized = False

    def _get_sparse_model(self) -> SparseTextEmbedding:
        """Lazily initialize the local FastEmbed BM25/SPLADE sparse vector model."""
        if self._sparse_model is None:
            logger.info("Loading SparseTextEmbedding model: %s", self.settings.sparse_model_name)
            self._sparse_model = SparseTextEmbedding(model_name=self.settings.sparse_model_name)
        return self._sparse_model

    async def ensure_collection(self) -> None:
        """
        Ensure the Qdrant collection exists with named Dense (Cosine) and Sparse (IDF)
        vector configurations plus keyword payload indexes for idempotency and filtering.
        """
        if self._collection_initialized:
            return

        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            logger.info("Creating hybrid Qdrant collection '%s'...", self.collection_name)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.settings.embedding_dimensions,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

            # Create payload indexes for fast O(1) hash idempotency checks and source file filtering
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_hash",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_file",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._collection_initialized = True

    async def check_file_hash_exists(self, file_hash: str) -> Tuple[bool, int]:
        """
        Idempotency check: verify whether a document with the given SHA-256 hash
        is already indexed in Qdrant.

        Returns:
            Tuple of (already_exists: bool, existing_chunk_count: int)
        """
        await self.ensure_collection()
        count_result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash",
                        match=models.MatchValue(value=file_hash),
                    )
                ]
            ),
            exact=True,
        )
        return (count_result.count > 0, count_result.count)

    async def generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense semantic embeddings via LiteLLM / OpenAI API.
        Falls back to a deterministic unit-norm hash embedding only if no API key is configured
        in local test/demo environments.
        """
        if not self.settings.openai_api_key or self.settings.openai_api_key.startswith("sk-your-"):
            logger.warning(
                "No valid OPENAI_API_KEY detected; using deterministic local dense vectors for offline/test mode."
            )
            return [self._deterministic_fallback_embedding(t) for t in texts]

        response = await litellm.aembedding(
            model=self.settings.embedding_model,
            input=texts,
            api_key=self.settings.openai_api_key,
        )
        return [item["embedding"] for item in response.data]

    def _deterministic_fallback_embedding(self, text: str) -> List[float]:
        """Generate a normalized deterministic dense vector when running offline or in tests."""
        dim = self.settings.embedding_dimensions
        vec = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                idx = int.from_bytes(digest[i : i + 2], "little") % dim
                sign = 1.0 if (digest[i + 2] % 2 == 0) else -1.0
                vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def generate_sparse_embeddings(self, texts: List[str]) -> List[models.SparseVector]:
        """Generate BM25 sparse vectors using FastEmbed in a background worker thread."""

        def _compute() -> List[models.SparseVector]:
            model = self._get_sparse_model()
            raw_embeddings = list(model.embed(texts))
            return [
                models.SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
                for emb in raw_embeddings
            ]

        return await asyncio.to_thread(_compute)

    async def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Generate both Dense and Sparse vectors concurrently in batches and upsert
        into Qdrant with full structural metadata.
        """
        if not chunks:
            return 0

        await self.ensure_collection()
        texts = [c.text for c in chunks]

        # Generate dense and sparse embeddings concurrently
        dense_vectors, sparse_vectors = await asyncio.gather(
            self.generate_dense_embeddings(texts),
            self.generate_sparse_embeddings(texts),
        )

        points: List[models.PointStruct] = []
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            payload = {
                "text": chunk.text,
                "source_file": chunk.metadata.source_file,
                "file_hash": chunk.metadata.file_hash,
                "page_number": chunk.metadata.page_number,
                "section_title": chunk.metadata.section_title,
                "chunk_index": chunk.metadata.chunk_index,
                "char_count": chunk.metadata.char_count,
                "ingested_at": chunk.metadata.ingested_at,
            }
            points.append(
                models.PointStruct(
                    id=chunk.point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=payload,
                )
            )

        # Upsert in batches of 64 points for optimal gRPC/HTTP payload sizing
        batch_size = 64
        for i in range(0, len(points), batch_size):
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
                wait=True,
            )

        return len(points)

    async def hybrid_search(
        self,
        query_text: str,
        source_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Perform Hybrid Search combining Dense semantic similarity and Sparse BM25 keyword
        matching via Qdrant's native Reciprocal Rank Fusion (RRF).

        Returns:
            Tuple of (fused_candidates, embedding_time_ms, qdrant_search_time_ms)
        """
        await self.ensure_collection()

        embed_start = time.perf_counter()
        dense_vecs, sparse_vecs = await asyncio.gather(
            self.generate_dense_embeddings([query_text]),
            self.generate_sparse_embeddings([query_text]),
        )
        dense_query = dense_vecs[0]
        sparse_query = sparse_vecs[0]
        embedding_ms = round((time.perf_counter() - embed_start) * 1000, 2)

        query_filter: Optional[models.Filter] = None
        if source_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=source_filter),
                    )
                ]
            )

        prefetch_limit = self.settings.hybrid_prefetch_limit
        rrf_limit = limit or self.settings.rrf_top_k

        search_start = time.perf_counter()
        response = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=rrf_limit,
            with_payload=True,
        )
        qdrant_search_ms = round((time.perf_counter() - search_start) * 1000, 2)

        candidates: List[Dict[str, Any]] = []
        for scored_point in response.points:
            payload = scored_point.payload or {}
            candidates.append(
                {
                    "point_id": str(scored_point.id),
                    "text": payload.get("text", ""),
                    "source_file": payload.get("source_file", "unknown"),
                    "file_hash": payload.get("file_hash", ""),
                    "page_number": int(payload.get("page_number", 1)),
                    "section_title": payload.get("section_title", "General"),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "rrf_score": round(float(scored_point.score or 0.0), 4),
                }
            )

        return candidates, embedding_ms, qdrant_search_ms

    async def list_ingested_documents(self) -> List[DocumentSummary]:
        """Aggregate and return metadata summaries of all documents currently stored in Qdrant."""
        await self.ensure_collection()
        records, _ = await self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        docs_by_hash: Dict[str, Dict[str, Any]] = {}
        for point in records:
            payload = point.payload or {}
            f_hash = payload.get("file_hash")
            if not f_hash:
                continue
            page_num = int(payload.get("page_number", 1))
            if f_hash not in docs_by_hash:
                docs_by_hash[f_hash] = {
                    "source_file": payload.get("source_file", "unknown"),
                    "file_hash": f_hash,
                    "chunk_count": 0,
                    "total_pages": page_num,
                    "ingested_at": payload.get("ingested_at"),
                }
            docs_by_hash[f_hash]["chunk_count"] += 1
            if page_num > docs_by_hash[f_hash]["total_pages"]:
                docs_by_hash[f_hash]["total_pages"] = page_num

        return [DocumentSummary(**doc) for doc in docs_by_hash.values()]
