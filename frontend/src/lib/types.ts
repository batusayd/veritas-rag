export interface IngestResponse {
  status: "ingested" | "duplicate_skipped";
  source_file: string;
  file_hash: string;
  total_pages: number;
  chunks_indexed: number;
  processing_time_ms: number;
  message: string;
}

export interface DocumentSummary {
  source_file: string;
  file_hash: string;
  chunk_count: number;
  total_pages: number;
  ingested_at?: string | null;
}

export interface Citation {
  source_file: string;
  page_number: number;
  section_title: string;
  chunk_index: number;
  exact_quote: string;
  relevance_score: number;
}

export interface LatencyBreakdown {
  query_rewrite_ms: number;
  embedding_ms: number;
  qdrant_search_ms: number;
  rerank_ms: number;
  llm_generation_ms: number;
  total_ms: number;
}

export interface RetrievedChunkPreview {
  point_id: string;
  source_file: string;
  page_number: number;
  section_title: string;
  chunk_index: number;
  text: string;
  rrf_score: number;
  rerank_score: number;
}

export interface QueryResponse {
  original_query: string;
  rewritten_query: string;
  answer: string;
  citations: Citation[];
  confidence_score: number;
  follow_up_questions: string[];
  refused: boolean;
  latency: LatencyBreakdown;
  retrieved_chunks: RetrievedChunkPreview[];
}

export interface ChatTurn {
  id: string;
  userQuery: string;
  response?: QueryResponse;
  error?: string;
  isLoading: boolean;
  timestamp: string;
}
