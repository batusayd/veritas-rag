import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.config import Settings
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_refusal_guardrail_when_no_chunks_pass_reranker(test_settings: Settings):
    llm = LLMService(settings=test_settings)
    output, elapsed_ms = await llm.generate_structured_answer(
        original_query="What is the secret launch code?",
        rewritten_query="secret launch code",
        reranked_chunks=[],
    )

    assert output.refused is True
    assert output.confidence_score == 0.0
    assert output.citations == []
    assert len(output.follow_up_questions) == 2
    assert elapsed_ms >= 0.0


@pytest.mark.asyncio
async def test_mocked_llm_structured_citation_parsing():
    live_settings = Settings(
        APP_ENV="test",
        OPENAI_API_KEY="sk-live-test-key-for-mock",
    )
    llm = LLMService(settings=live_settings)

    reranked_chunks = [
        {
            "point_id": "id-1",
            "source_file": "system_spec.pdf",
            "page_number": 4,
            "section_title": "Retrieval Architecture",
            "chunk_index": 7,
            "text": "Qdrant executes Reciprocal Rank Fusion (RRF) over dense cosine and sparse BM25 prefetch lists.",
            "rrf_score": 0.78,
            "rerank_score": 0.9412,
        }
    ]

    mock_llm_json = json.dumps(
        {
            "answer": "Qdrant performs **Reciprocal Rank Fusion (RRF)** combining dense cosine and sparse BM25 lists `[system_spec.pdf, p. 4]`.",
            "citations": [
                {
                    "source_file": "system_spec.pdf",
                    "page_number": 4,
                    "section_title": "Retrieval Architecture",
                    "chunk_index": 7,
                    "exact_quote": "Qdrant executes Reciprocal Rank Fusion (RRF) over dense cosine and sparse BM25 prefetch lists.",
                    "relevance_score": 0.5,
                }
            ],
            "confidence_score": 0.95,
            "follow_up_questions": [
                "How many candidates are prefetched per vector modality?",
                "How does FlashRank score the RRF-fused results?",
            ],
            "refused": False,
        }
    )

    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(message=MagicMock(content=mock_llm_json))
    ]

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_completion_response)):
        structured, llm_ms = await llm.generate_structured_answer(
            original_query="How does Qdrant combine dense and sparse results?",
            rewritten_query="Qdrant Reciprocal Rank Fusion dense cosine sparse BM25",
            reranked_chunks=reranked_chunks,
        )

    assert structured.refused is False
    assert structured.confidence_score == pytest.approx(0.95)
    assert len(structured.citations) == 1
    assert structured.citations[0].source_file == "system_spec.pdf"
    assert structured.citations[0].page_number == 4
    # Verify relevance_score was enriched from the Cross-Encoder reranked chunk (0.9412)
    assert structured.citations[0].relevance_score == pytest.approx(0.9412)
    assert len(structured.follow_up_questions) == 2
    assert llm_ms >= 0.0
