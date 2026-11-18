import hashlib
import re
import uuid
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

from app.config import get_settings
from app.models.schemas import ChunkMetadata, DocumentChunk

# Namespace UUID for deterministic point ID generation (file_hash + chunk_index)
CHUNK_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class DocumentParser:
    """
    Parses PDF, Markdown, and TXT documents into semantically coherent chunks
    with configurable overlap and rich structural metadata (page number, section title, file hash).
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        # Hierarchical separators for recursive semantic splitting
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Compute SHA-256 hex digest of raw file bytes for idempotent ingestion."""
        return hashlib.sha256(file_bytes).hexdigest()

    @staticmethod
    def generate_point_id(file_hash: str, chunk_index: int) -> str:
        """Generate a deterministic UUID5 from file_hash and chunk_index."""
        return str(uuid.uuid5(CHUNK_UUID_NAMESPACE, f"{file_hash}:{chunk_index}"))

    def parse_document(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Tuple[str, int, List[DocumentChunk]]:
        """
        Parse raw file bytes into structured DocumentChunk objects.

        Returns:
            Tuple of (file_hash, total_pages, chunks)
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        file_hash = self.compute_file_hash(file_bytes)

        if ext == ".pdf":
            pages_data = self._extract_pdf_pages(file_bytes)
        else:
            text_content = file_bytes.decode("utf-8", errors="replace")
            pages_data = [(1, text_content)]

        total_pages = max(len(pages_data), 1)
        chunks: List[DocumentChunk] = []
        chunk_index = 0
        current_section = "Introduction"

        for page_number, page_text in pages_data:
            cleaned_page = self._clean_text(page_text)
            if not cleaned_page:
                continue

            # Split page into section-aware blocks before recursive chunking
            section_blocks = self._extract_sections(cleaned_page, current_section)

            for section_title, block_text in section_blocks:
                current_section = section_title
                text_chunks = self._recursive_split(block_text)

                for raw_chunk in text_chunks:
                    normalized = raw_chunk.strip()
                    if not normalized:
                        continue

                    point_id = self.generate_point_id(file_hash, chunk_index)
                    metadata = ChunkMetadata(
                        source_file=filename,
                        file_hash=file_hash,
                        page_number=page_number,
                        section_title=section_title,
                        chunk_index=chunk_index,
                        char_count=len(normalized),
                    )
                    chunks.append(
                        DocumentChunk(
                            point_id=point_id,
                            text=normalized,
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1

        if not chunks:
            raise ValueError(f"No extractable text found in '{filename}'.")

        return file_hash, total_pages, chunks

    def _extract_pdf_pages(self, file_bytes: bytes) -> List[Tuple[int, str]]:
        """Extract text page-by-page from a PDF byte stream using PyMuPDF (fitz)."""
        pages: List[Tuple[int, str]] = []
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
            for idx, page in enumerate(pdf_doc, start=1):
                text = page.get_text("text")
                if text and text.strip():
                    pages.append((idx, text))
        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace while preserving paragraph and heading boundaries."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove null bytes and control characters except newline/tab
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse 3+ consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces/tabs on a single line
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _extract_sections(text: str, default_section: str) -> List[Tuple[str, str]]:
        """
        Detect Markdown headings (# Heading) or numbered uppercase PDF headings
        to preserve section_title metadata across chunks.
        """
        lines = text.split("\n")
        sections: List[Tuple[str, str]] = []
        current_title = default_section
        buffer: List[str] = []

        md_heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$")
        pdf_heading_pattern = re.compile(
            r"^(?:(?:\d+\.)+\d*|\b(?:SECTION|CHAPTER|PART)\s+\d+)\s+[A-Z][A-Za-z0-9\s,&:-]{2,65}$"
        )

        for line in lines:
            stripped = line.strip()
            md_match = md_heading_pattern.match(stripped)
            is_pdf_heading = bool(pdf_heading_pattern.match(stripped))

            if md_match or is_pdf_heading:
                if buffer:
                    combined = "\n".join(buffer).strip()
                    if combined:
                        sections.append((current_title, combined))
                    buffer = []
                current_title = (md_match.group(2) if md_match else stripped).strip()[:100]
                buffer.append(stripped)
            else:
                buffer.append(line)

        if buffer:
            combined = "\n".join(buffer).strip()
            if combined:
                sections.append((current_title, combined))

        return sections or [(default_section, text)]

    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using semantic boundaries (paragraphs -> lines -> sentences -> words)
        and apply sliding-window overlap to preserve context continuity across chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        atomic_segments = self._split_with_separators(text, self.separators)
        return self._merge_segments_with_overlap(atomic_segments)

    def _split_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """Recursively break text down until all pieces are <= chunk_size."""
        if len(text) <= self.chunk_size or not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Hard character fallback if a single token exceeds chunk_size
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        parts = text.split(separator)
        results: List[str] = []
        for idx, part in enumerate(parts):
            if not part:
                continue
            # Re-attach punctuation/newline separator to preserve natural reading structure
            piece = part + (separator if idx < len(parts) - 1 and separator.strip() else "")
            if len(piece) <= self.chunk_size:
                results.append(piece)
            else:
                results.extend(self._split_with_separators(piece, remaining_separators))
        return results

    def _merge_segments_with_overlap(self, segments: List[str]) -> List[str]:
        """Merge small segments into chunks up to chunk_size with sliding overlap."""
        chunks: List[str] = []
        current_segments: List[str] = []
        current_length = 0

        for seg in segments:
            seg_clean = seg.strip()
            if not seg_clean:
                continue
            seg_len = len(seg_clean)

            if current_length + seg_len + (1 if current_segments else 0) > self.chunk_size:
                if current_segments:
                    chunks.append(" ".join(current_segments))

                    # Retain trailing segments up to chunk_overlap for continuity
                    overlap_segments: List[str] = []
                    overlap_len = 0
                    for prev_seg in reversed(current_segments):
                        if overlap_len + len(prev_seg) <= self.chunk_overlap:
                            overlap_segments.insert(0, prev_seg)
                            overlap_len += len(prev_seg) + 1
                        else:
                            break

                    current_segments = overlap_segments
                    current_length = sum(len(s) for s in current_segments) + max(
                        0, len(current_segments) - 1
                    )

            current_segments.append(seg_clean)
            current_length += seg_len + (1 if len(current_segments) > 1 else 0)

        if current_segments:
            chunks.append(" ".join(current_segments))

        return chunks
