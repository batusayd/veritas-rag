import fitz
import pytest

from app.services.parser import DocumentParser


def test_compute_file_hash_and_point_id_are_deterministic(parser: DocumentParser):
    raw_bytes = b"# Hybrid RAG Architecture\nDeterministic hashing test."
    hash_1 = parser.compute_file_hash(raw_bytes)
    hash_2 = parser.compute_file_hash(raw_bytes)

    assert hash_1 == hash_2
    assert len(hash_1) == 64

    point_id_1 = parser.generate_point_id(hash_1, 0)
    point_id_2 = parser.generate_point_id(hash_1, 0)
    point_id_next = parser.generate_point_id(hash_1, 1)

    assert point_id_1 == point_id_2
    assert point_id_1 != point_id_next


def test_parse_markdown_preserves_sections_and_overlap(parser: DocumentParser):
    markdown_content = (
        "# Vector Indexing\n"
        "Qdrant stores both dense semantic embeddings and sparse BM25 vectors in a single collection. "
        "This dual-vector design enables high-recall retrieval across both conceptual queries and exact identifiers.\n\n"
        "## Cross-Encoder Reranking\n"
        "FlashRank evaluates query-document pairs jointly using a cross-encoder transformer to filter noise "
        "before sending top-K context chunks to the language model."
    ).encode("utf-8")

    file_hash, total_pages, chunks = parser.parse_document(
        file_bytes=markdown_content,
        filename="rag_design.md",
    )

    assert total_pages == 1
    assert len(chunks) >= 2
    assert chunks[0].metadata.source_file == "rag_design.md"
    assert chunks[0].metadata.file_hash == file_hash
    assert chunks[0].metadata.section_title == "Vector Indexing"
    assert any(c.metadata.section_title == "Cross-Encoder Reranking" for c in chunks)

    # Verify sequential chunk_index and character bounds
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == idx
        assert len(chunk.text) <= parser.chunk_size


def test_parse_in_memory_pdf_preserves_page_numbers(parser: DocumentParser):
    pdf_doc = fitz.open()
    page1 = pdf_doc.new_page()
    page1.insert_text((72, 72), "1.0 INTRODUCTION\nPage one covers dense vector indexing.")
    page2 = pdf_doc.new_page()
    page2.insert_text((72, 72), "2.0 SPARSE RETRIEVAL\nPage two covers BM25 lexical matching.")
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    _, total_pages, chunks = parser.parse_document(
        file_bytes=pdf_bytes,
        filename="whitepaper.pdf",
    )

    assert total_pages == 2
    page_numbers = {c.metadata.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_unsupported_extension_raises_value_error(parser: DocumentParser):
    with pytest.raises(ValueError, match="Unsupported file type"):
        parser.parse_document(b"binary data", "malware.exe")
