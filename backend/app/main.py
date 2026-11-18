from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import get_qdrant_service, router
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Qdrant hybrid collection on startup."""
    qdrant = get_qdrant_service()
    try:
        await qdrant.ensure_collection()
        logger.info("Qdrant hybrid collection '%s' ready.", qdrant.collection_name)
    except Exception as exc:
        logger.warning("Qdrant startup check deferred (will retry on request): %s", exc)
    yield


app = FastAPI(
    title="VeritasRAG — Hybrid Document Intelligence API",
    description=(
        "Production-grade Hybrid RAG Engine combining Dense Embeddings + Sparse BM25 in Qdrant, "
        "Reciprocal Rank Fusion (RRF), FlashRank Cross-Encoder Reranking, and Strict Pydantic Citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
