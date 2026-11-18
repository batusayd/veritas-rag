import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import litellm

from app.config import Settings, get_settings
from app.models.schemas import Citation, LLMStructuredOutput

logger = logging.getLogger(__name__)

QUERY_REWRITE_SYSTEM_PROMPT = """You are a search query optimization specialist for an enterprise Hybrid RAG engine (Dense Vector + BM25 Keyword Search).
Rewrite the user's query to maximize retrieval precision and recall:
1. Preserve exact technical identifiers, acronyms, error codes, and proper nouns for BM25 keyword matching.
2. Expand ambiguous phrasing into clear, self-contained semantic search terms.
3. Output ONLY the rewritten search query string with no quotes, preamble, or explanation."""

RAG_GENERATION_SYSTEM_PROMPT = """You are a rigorous Document Intelligence Analyst.
Answer the user's question ONLY using the provided [Retrieved Context Chunks].

CRITICAL RULES:
1. GROUNDING & REFUSAL: If the retrieved context does not contain sufficient evidence to answer the question, you MUST set "refused": true, "confidence_score": 0.0, "citations": [], and clearly state in "answer" that the provided documents do not contain enough information to answer the question. Never extrapolate or hallucinate facts outside the context.
2. EXACT CITATIONS: Every claim in your answer must be backed by a citation entry in "citations". Each citation MUST include:
   - "source_file": Exact file name from the chunk header
   - "page_number": Exact integer page number from the chunk header
   - "section_title": Exact section title from the chunk header
   - "chunk_index": Exact integer chunk index from the chunk header
   - "exact_quote": A verbatim substring copied directly from that chunk's text
   - "relevance_score": Float between 0.0 and 1.0
3. MARKDOWN FORMATTING: Format "answer" in clean Markdown using bullet points, bold terms, and inline references like `[source_file, p. X]`.
4. FOLLOW-UP QUESTIONS: Provide exactly 2 insightful follow-up questions in "follow_up_questions" that explore related topics covered in the context.
5. STRICT JSON: Respond with valid JSON conforming strictly to the requested schema."""


class LLMService:
    """
    Manages LLM Query Rewriting and Hallucination-Free Structured JSON Generation
    with Pydantic v2 validation and strict citation enforcement.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def _has_live_api_key(self) -> bool:
        key = self.settings.openai_api_key or self.settings.anthropic_api_key
        return bool(key and not key.startswith("sk-your-") and not key.startswith("sk-ant-your-"))

    async def rewrite_query(self, original_query: str) -> Tuple[str, float]:
        """
        Optimize ambiguous user queries for hybrid dense + sparse retrieval.

        Returns:
            Tuple of (rewritten_query, rewrite_latency_ms)
        """
        start_time = time.perf_counter()
        cleaned = original_query.strip()

        if not self._has_live_api_key():
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return cleaned, elapsed_ms

        try:
            response = await litellm.acompletion(
                model=self.settings.query_rewriter_model,
                messages=[
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned},
                ],
                temperature=0.0,
                max_tokens=120,
                api_key=self.settings.openai_api_key,
            )
            rewritten = (
                response.choices[0].message.content.strip().strip('"').strip("'")
                or cleaned
            )
        except Exception as exc:
            logger.warning("Query rewrite failed, falling back to original query: %s", exc)
            rewritten = cleaned

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return rewritten, elapsed_ms

    async def generate_structured_answer(
        self,
        original_query: str,
        rewritten_query: str,
        reranked_chunks: List[Dict[str, Any]],
    ) -> Tuple[LLMStructuredOutput, float]:
        """
        Generate a strictly typed LLMStructuredOutput grounded in reranked chunks.
        Explicitly refuses to answer if no relevant context chunks survive reranking.
        """
        start_time = time.perf_counter()

        # Hard guardrail: if no chunks passed the cross-encoder relevance threshold, refuse immediately
        if not reranked_chunks:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            refusal = LLMStructuredOutput(
                answer=(
                    "I cannot answer this question because no relevant information was found "
                    "in the indexed documents that meets the relevance threshold."
                ),
                citations=[],
                confidence_score=0.0,
                follow_up_questions=[
                    "Which documents are currently indexed in the knowledge base?",
                    "Can you upload a document covering this topic and try again?",
                ],
                refused=True,
            )
            return refusal, elapsed_ms

        formatted_context = self._format_context_blocks(reranked_chunks)

        if not self._has_live_api_key():
            # Deterministic offline synthesis for local verification without API keys
            output = self._offline_extractive_fallback(original_query, reranked_chunks)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return output, elapsed_ms

        user_prompt = (
            f"User Question: {original_query}\n"
            f"Optimized Search Query: {rewritten_query}\n\n"
            f"[Retrieved Context Chunks]\n{formatted_context}\n\n"
            "Return ONLY a JSON object matching this exact schema:\n"
            "{\n"
            '  "answer": "markdown string",\n'
            '  "citations": [\n'
            "    {\n"
            '      "source_file": "string",\n'
            '      "page_number": 1,\n'
            '      "section_title": "string",\n'
            '      "chunk_index": 0,\n'
            '      "exact_quote": "verbatim quote from chunk",\n'
            '      "relevance_score": 0.95\n'
            "    }\n"
            "  ],\n"
            '  "confidence_score": 0.92,\n'
            '  "follow_up_questions": ["question 1?", "question 2?"],\n'
            '  "refused": false\n'
            "}"
        )

        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": RAG_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            api_key=self.settings.openai_api_key,
        )

        raw_json = response.choices[0].message.content or "{}"
        parsed = self._parse_and_enrich_output(raw_json, reranked_chunks)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return parsed, elapsed_ms

    @staticmethod
    def _format_context_blocks(chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks with explicit metadata headers for accurate LLM citation."""
        blocks: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            header = (
                f"--- CHUNK #{idx} | "
                f"source_file: {chunk['source_file']} | "
                f"page_number: {chunk['page_number']} | "
                f"section_title: {chunk.get('section_title', 'General')} | "
                f"chunk_index: {chunk['chunk_index']} | "
                f"rerank_score: {chunk.get('rerank_score', 0.0):.4f} ---"
            )
            blocks.append(f"{header}\n{chunk['text']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _parse_and_enrich_output(
        raw_json: str,
        reranked_chunks: List[Dict[str, Any]],
    ) -> LLMStructuredOutput:
        """
        Validate raw LLM JSON against LLMStructuredOutput and attach cross-encoder
        relevance scores from the retrieved chunks.
        """
        chunk_score_lookup = {
            (c["source_file"], int(c["chunk_index"])): float(c.get("rerank_score", 0.0))
            for c in reranked_chunks
        }
        payload = json.loads(raw_json)

        # Ensure follow_up_questions always has exactly 2 items
        follow_ups = payload.get("follow_up_questions") or []
        while len(follow_ups) < 2:
            follow_ups.append("What other details are mentioned in this section?")
        payload["follow_up_questions"] = follow_ups[:2]

        for citation in payload.get("citations", []):
            key = (citation.get("source_file", ""), int(citation.get("chunk_index", 0)))
            if key in chunk_score_lookup:
                citation["relevance_score"] = round(min(1.0, max(0.0, chunk_score_lookup[key])), 4)

        return LLMStructuredOutput.model_validate(payload)

    @staticmethod
    def _offline_extractive_fallback(
        query: str,
        reranked_chunks: List[Dict[str, Any]],
    ) -> LLMStructuredOutput:
        """Extractive grounded fallback when running without an external LLM API key."""
        top_chunk = reranked_chunks[0]
        quote = top_chunk["text"][:280].strip()
        score = round(min(1.0, max(0.0, float(top_chunk.get("rerank_score", 0.75)))), 4)

        citations = [
            Citation(
                source_file=c["source_file"],
                page_number=int(c["page_number"]),
                section_title=c.get("section_title", "General"),
                chunk_index=int(c["chunk_index"]),
                exact_quote=c["text"][:240].strip(),
                relevance_score=round(min(1.0, max(0.0, float(c.get("rerank_score", 0.7)))), 4),
            )
            for c in reranked_chunks[:2]
        ]

        answer = (
            f"Based on **{top_chunk['source_file']}** (Page {top_chunk['page_number']}, "
            f"*{top_chunk.get('section_title', 'General')}*):\n\n"
            f"> \"{quote}\"\n\n"
            f"*(Synthesized in offline extractive mode — configure `OPENAI_API_KEY` for full generative synthesis.)*"
        )

        return LLMStructuredOutput(
            answer=answer,
            citations=citations,
            confidence_score=score,
            follow_up_questions=[
                f"What else does {top_chunk['source_file']} state on page {top_chunk['page_number']}?",
                f"Are there additional sections related to {top_chunk.get('section_title', 'this topic')}?",
            ],
            refused=False,
        )
