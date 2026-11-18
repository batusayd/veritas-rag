import logging
import time
from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.models.schemas import (
    DocumentSummary,
    IngestResponse,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    RetrievedChunkPreview,
)
from app.services.llm_service import LLMService
from app.services.parser import DocumentParser
from app.services.qdrant_service import QdrantHybridService
from app.services.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Hybrid RAG Intelligence"])


@lru_cache
def get_parser() -> DocumentParser:
    return DocumentParser()


@lru_cache
def get_qdrant_service() -> QdrantHybridService:
    return QdrantHybridService()


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(
    qdrant: QdrantHybridService = Depends(get_qdrant_service),
) -> dict:
    """Verify API readiness and Qdrant connectivity."""
    try:
        await qdrant.ensure_collection()
        return {
            "status": "healthy",
            "collection": qdrant.collection_name,
            "hybrid_vectors": ["dense", "sparse (BM25)"],
        }
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant connection failed: {exc}",
        ) from exc


@router.get("/documents", response_model=List[DocumentSummary])
async def list_documents(
    qdrant: QdrantHybridService = Depends(get_qdrant_service),
) -> List[DocumentSummary]:
    """List all unique documents currently indexed in Qdrant."""
    return await qdrant.list_ingested_documents()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, Markdown (.md), or Text (.txt) file"),
    parser: DocumentParser = Depends(get_parser),
    qdrant: QdrantHybridService = Depends(get_qdrant_service),
) -> IngestResponse:
    """
    Advanced Ingestion Pipeline:
    1. Read uploaded file bytes and compute SHA-256 hash.
    2. Check Qdrant payload index for idempotency (skip if already ingested).
    3. Parse document (PyMuPDF for PDF, native UTF-8 for MD/TXT) into recursive semantic chunks
       preserving `source_file`, `page_number`, `section_title`, and `chunk_index`.
    4. Generate Dense + Sparse (BM25) vectors concurrently and upsert into Qdrant.
    """
    start_time = time.perf_counter()
    filename = file.filename or "untitled.txt"

    try:
        file_bytes = await file.read()
        file_hash = parser.compute_file_hash(file_bytes)

        # Step 1: Idempotency check in Qdrant before parsing/embedding
        already_exists, existing_count = await qdrant.check_file_hash_exists(file_hash)
        if already_exists:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "Skipping duplicate upload '%s' (hash=%s, chunks=%d)",
                filename,
                file_hash[:12],
                existing_count,
            )
            return IngestResponse(
                status="duplicate_skipped",
                source_file=filename,
                file_hash=file_hash,
                total_pages=1,
                chunks_indexed=existing_count,
                processing_time_ms=elapsed_ms,
                message=(
                    f"Document '{filename}' (SHA-256: {file_hash[:12]}...) is already indexed "
                    f"with {existing_count} chunks. Duplicate ingestion skipped."
                ),
            )

        # Step 2: Parse & chunk document with overlap and metadata preservation
        file_hash, total_pages, chunks = parser.parse_document(
            file_bytes=file_bytes,
            filename=filename,
        )

        # Step 3: Generate Hybrid (Dense + Sparse BM25) embeddings and upsert into Qdrant
        indexed_count = await qdrant.index_chunks(chunks)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Ingested '%s': pages=%d, chunks=%d, time=%.2fms",
            filename,
            total_pages,
            indexed_count,
            elapsed_ms,
        )

        return IngestResponse(
            status="ingested",
            source_file=filename,
            file_hash=file_hash,
            total_pages=total_pages,
            chunks_indexed=indexed_count,
            processing_time_ms=elapsed_ms,
            message=f"Successfully parsed and indexed {indexed_count} hybrid chunks from '{filename}'.",
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.exception("Failed to ingest document '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(
    request: QueryRequest,
    qdrant: QdrantHybridService = Depends(get_qdrant_service),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    llm: LLMService = Depends(get_llm_service),
) -> QueryResponse:
    """
    Hybrid Retrieval, Cross-Encoder Reranking & Hallucination-Free Structured Synthesis:
    1. Rewrite/optimize query via fast LLM.
    2. Execute Hybrid Search (Dense Cosine + Sparse BM25) with Reciprocal Rank Fusion (RRF) in Qdrant.
    3. Rerank top RRF candidates with FlashRank Cross-Encoder and filter below score threshold.
    4. Synthesize strictly typed Pydantic response with verbatim citations and latency breakdown.
    """
    pipeline_start = time.perf_counter()

    try:
        # Stage 1: Query Rewriting
        rewritten_query, rewrite_ms = await llm.rewrite_query(request.query)

        # Stage 2: Qdrant Hybrid Search (Dense + Sparse BM25 -> RRF)
        rrf_candidates, embedding_ms, qdrant_search_ms = await qdrant.hybrid_search(
            query_text=rewritten_query,
            source_filter=request.source_filter,
        )

        # Stage 3: FlashRank Cross-Encoder Reranking
        reranked_chunks, rerank_ms = await reranker.rerank(
            query=rewritten_query,
            candidates=rrf_candidates,
            top_k=request.top_k,
        )

        # Stage 4: Structured LLM Generation with Strict Citations
        structured_output, llm_ms = await llm.generate_structured_answer(
            original_query=request.query,
            rewritten_query=rewritten_query,
            reranked_chunks=reranked_chunks,
        )

        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

        latency = LatencyBreakdown(
            query_rewrite_ms=rewrite_ms,
            embedding_ms=embedding_ms,
            qdrant_search_ms=qdrant_search_ms,
            rerank_ms=rerank_ms,
            llm_generation_ms=llm_ms,
            total_ms=total_ms,
        )

        previews = [
            RetrievedChunkPreview(
                point_id=c["point_id"],
                source_file=c["source_file"],
                page_number=int(c["page_number"]),
                section_title=c.get("section_title", "General"),
                chunk_index=int(c["chunk_index"]),
                text=c["text"],
                rrf_score=float(c.get("rrf_score", 0.0)),
                rerank_score=float(c.get("rerank_score", 0.0)),
            )
            for c in reranked_chunks
        ]

        logger.info(
            "Query completed in %.2fms (rewrite=%.1fms, embed=%.1fms, qdrant=%.1fms, rerank=%.1fms, llm=%.1fms)",
            total_ms,
            rewrite_ms,
            embedding_ms,
            qdrant_search_ms,
            rerank_ms,
            llm_ms,
        )

        return QueryResponse(
            original_query=request.query,
            rewritten_query=rewritten_query,
            answer=structured_output.answer,
            citations=structured_output.citations,
            confidence_score=structured_output.confidence_score,
            follow_up_questions=structured_output.follow_up_questions,
            refused=structured_output.refused,
            latency=latency,
            retrieved_chunks=previews,
        )

    except Exception as exc:
        logger.exception("Query pipeline failed for query='%s'", request.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid RAG query pipeline failed: {exc}",
        ) from exc
