import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from flashrank import Ranker, RerankRequest

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Lightweight local Cross-Encoder Reranker powered by FlashRank (ms-marco-MiniLM-L-12-v2).
    Scores and filters RRF-fused hybrid search candidates before passing top-K chunks to the LLM.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._ranker: Optional[Ranker] = None

    def _get_ranker(self) -> Optional[Ranker]:
        """Lazily load the local ONNX Cross-Encoder model."""
        if self._ranker is None:
            try:
                logger.info("Initializing FlashRank Cross-Encoder: %s", self.settings.reranker_model_name)
                self._ranker = Ranker(model_name=self.settings.reranker_model_name)
            except Exception as exc:
                logger.warning("FlashRank initialization fallback triggered: %s", exc)
                self._ranker = None
        return self._ranker

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Rerank candidate chunks using a Cross-Encoder and filter out low-relevance results.

        Returns:
            Tuple of (filtered_top_k_chunks, rerank_time_ms)
        """
        start_time = time.perf_counter()
        if not candidates:
            return [], 0.0

        limit = top_k or self.settings.rerank_top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self.settings.rerank_score_threshold
        )

        def _run_cross_encoder() -> List[Dict[str, Any]]:
            ranker = self._get_ranker()
            if ranker is not None:
                passages = [
                    {
                        "id": idx,
                        "text": item["text"],
                        "meta": item,
                    }
                    for idx, item in enumerate(candidates)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                ranked_results = ranker.rerank(rerank_req)

                scored_chunks: List[Dict[str, Any]] = []
                for res in ranked_results:
                    score = round(float(res.get("score", 0.0)), 4)
                    if score < threshold:
                        continue
                    meta = dict(res["meta"])
                    meta["rerank_score"] = score
                    scored_chunks.append(meta)

                scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_chunks[:limit]

            # Deterministic lexical overlap + RRF fallback if offline without cached ONNX weights
            query_terms = set(query.lower().split())
            fallback_chunks: List[Dict[str, Any]] = []
            for item in candidates:
                chunk_terms = set(item["text"].lower().split())
                overlap = (
                    len(query_terms & chunk_terms) / max(len(query_terms), 1)
                    if query_terms
                    else 0.0
                )
                combined_score = round(min(1.0, 0.5 * overlap + 0.5 * float(item.get("rrf_score", 0.5))), 4)
                if combined_score >= threshold:
                    enriched = dict(item)
                    enriched["rerank_score"] = combined_score
                    fallback_chunks.append(enriched)

            fallback_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            return fallback_chunks[:limit]

        reranked = await asyncio.to_thread(_run_cross_encoder)
        rerank_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return reranked, rerank_ms
