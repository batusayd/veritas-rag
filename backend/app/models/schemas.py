from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Rich metadata preserved for every semantic chunk."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(..., description="Original filename of the ingested document.")
    file_hash: str = Field(..., description="SHA-256 hex digest of raw document bytes.")
    page_number: int = Field(..., ge=1, description="1-based page number (1 for MD/TXT).")
    section_title: str = Field(
        default="General",
        description="Nearest detected heading or section title.",
    )
    chunk_index: int = Field(..., ge=0, description="0-based sequential chunk index within the file.")
    char_count: int = Field(..., ge=1, description="Character length of the chunk text.")
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of ingestion.",
    )


class DocumentChunk(BaseModel):
    """Internal representation of a parsed document chunk prior to vectorization."""

    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(..., description="Deterministic UUID5 generated from file_hash and chunk_index.")
    text: str = Field(..., min_length=1, description="Cleaned semantic chunk content.")
    metadata: ChunkMetadata


class IngestResponse(BaseModel):
    """Response returned by POST /api/ingest."""

    status: Literal["ingested", "duplicate_skipped"] = Field(
        ...,
        description="Whether the document was newly indexed or skipped due to idempotency.",
    )
    source_file: str
    file_hash: str
    total_pages: int = Field(..., ge=1)
    chunks_indexed: int = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0.0)
    message: str


class DocumentSummary(BaseModel):
    """Summary of an ingested document stored in Qdrant."""

    source_file: str
    file_hash: str
    chunk_count: int
    total_pages: int
    ingested_at: Optional[str] = None


class Citation(BaseModel):
    """Strictly typed source citation linked to a specific retrieved chunk."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(..., description="Exact filename of the cited source document.")
    page_number: int = Field(..., ge=1, description="1-based page number where the quote appears.")
    section_title: str = Field(default="General", description="Section title of the cited chunk.")
    chunk_index: int = Field(..., ge=0, description="Chunk index within the source document.")
    exact_quote: str = Field(
        ...,
        description="Verbatim quote extracted from the retrieved context supporting the answer.",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cross-encoder reranker relevance score for this source.",
    )


class LLMStructuredOutput(BaseModel):
    """Strict schema enforced on the LLM generation step."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description=(
            "Comprehensive Markdown response grounded strictly in the provided context. "
            "If the context does not contain enough information, explicitly refuse to answer."
        ),
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of exact quotes, file names, and page numbers supporting the answer.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) reflecting how well the context answers the query.",
    )
    follow_up_questions: List[str] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly 2 suggested follow-up questions relevant to the document corpus.",
    )
    refused: bool = Field(
        default=False,
        description="True if the system refused to answer due to insufficient context.",
    )


class LatencyBreakdown(BaseModel):
    """Stage-by-stage observability metrics in milliseconds."""

    query_rewrite_ms: float = Field(..., ge=0.0)
    embedding_ms: float = Field(..., ge=0.0)
    qdrant_search_ms: float = Field(..., ge=0.0)
    rerank_ms: float = Field(..., ge=0.0)
    llm_generation_ms: float = Field(..., ge=0.0)
    total_ms: float = Field(..., ge=0.0)


class RetrievedChunkPreview(BaseModel):
    """Preview of retrieved & reranked chunks returned for inspection in the UI."""

    point_id: str
    source_file: str
    page_number: int
    section_title: str
    chunk_index: int
    text: str
    rrf_score: float
    rerank_score: float


class QueryRequest(BaseModel):
    """Payload for POST /api/query."""

    query: str = Field(..., min_length=2, max_length=2000, description="User natural language question.")
    source_filter: Optional[str] = Field(
        default=None,
        description="Optional filename filter to restrict retrieval to a single document.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=15,
        description="Optional override for number of reranked chunks passed to the LLM.",
    )


class QueryResponse(BaseModel):
    """Full response returned by POST /api/query including structured answer and telemetry."""

    original_query: str
    rewritten_query: str
    answer: str
    citations: List[Citation]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    follow_up_questions: List[str]
    refused: bool
    latency: LatencyBreakdown
    retrieved_chunks: List[RetrievedChunkPreview] = Field(default_factory=list)
