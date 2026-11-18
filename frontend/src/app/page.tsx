"use client";

import React, { useEffect, useState } from "react";
import { Cpu, Database, Layers } from "lucide-react";
import ChatPanel from "@/components/ChatPanel";
import DocumentUploader from "@/components/DocumentUploader";
import SourceViewerPanel from "@/components/SourceViewerPanel";
import { fetchDocuments, queryKnowledgeBase } from "@/lib/api";
import {
  ChatTurn,
  Citation,
  DocumentSummary,
  IngestResponse,
  RetrievedChunkPreview,
} from "@/lib/types";

export default function HomePage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedFilter, setSelectedFilter] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(
    null
  );
  const [activeTurnChunks, setActiveTurnChunks] = useState<
    RetrievedChunkPreview[]
  >([]);

  const loadDocuments = async () => {
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
    } catch {
      // Backend may still be starting up
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleIngestSuccess = (_result: IngestResponse) => {
    loadDocuments();
  };

  const handleSendQuery = async (queryText: string) => {
    const turnId = `turn-${Date.now()}`;
    const newTurn: ChatTurn = {
      id: turnId,
      userQuery: queryText,
      isLoading: true,
      timestamp: new Date().toISOString(),
    };

    setTurns((prev) => [...prev, newTurn]);
    setIsSubmitting(true);

    try {
      const response = await queryKnowledgeBase(queryText, selectedFilter);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId ? { ...t, isLoading: false, response } : t
        )
      );

      setActiveTurnChunks(response.retrieved_chunks);
      if (response.citations.length > 0) {
        setSelectedCitation(response.citations[0]);
      } else {
        setSelectedCitation(null);
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Query failed";
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId ? { ...t, isLoading: false, error: errMsg } : t
        )
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSelectCitation = (
    citation: Citation,
    turnChunks: RetrievedChunkPreview[]
  ) => {
    setSelectedCitation(citation);
    setActiveTurnChunks(turnChunks);
  };

  const handleSelectChunkAsCitation = (chunk: RetrievedChunkPreview) => {
    setSelectedCitation({
      source_file: chunk.source_file,
      page_number: chunk.page_number,
      section_title: chunk.section_title,
      chunk_index: chunk.chunk_index,
      exact_quote: chunk.text.slice(0, 220),
      relevance_score: chunk.rerank_score,
    });
  };

  return (
    <div className="flex h-screen flex-col bg-slate-950">
      {/* Top Navbar */}
      <header className="flex items-center justify-between border-b border-slate-800/80 bg-slate-950 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600/20 text-indigo-400 ring-1 ring-indigo-500/30">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-slate-100">
              VeritasRAG · Document Intelligence Engine
            </h1>
            <p className="text-[11px] text-slate-400">
              Dense + Sparse BM25 · Qdrant RRF · FlashRank Cross-Encoder · Strict Pydantic Citations
            </p>
          </div>
        </div>

        <div className="hidden items-center gap-2 md:flex">
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300">
            <Database className="h-3.5 w-3.5 text-emerald-400" />
            Qdrant Hybrid (RRF)
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300">
            <Cpu className="h-3.5 w-3.5 text-amber-400" />
            FlashRank Cross-Encoder
          </span>
        </div>
      </header>

      {/* Document Upload & Filter Bar */}
      <DocumentUploader
        documents={documents}
        selectedFilter={selectedFilter}
        onSelectFilter={setSelectedFilter}
        onIngestSuccess={handleIngestSuccess}
      />

      {/* Split-Screen Workspace: Chat Left (55%) / Citation & Source Viewer Right (45%) */}
      <main className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-12">
        <section className="min-h-0 lg:col-span-7">
          <ChatPanel
            turns={turns}
            selectedCitation={selectedCitation}
            onSelectCitation={handleSelectCitation}
            onSendQuery={handleSendQuery}
            isSubmitting={isSubmitting}
          />
        </section>

        <section className="min-h-0 lg:col-span-5">
          <SourceViewerPanel
            selectedCitation={selectedCitation}
            retrievedChunks={activeTurnChunks}
            onSelectChunkAsCitation={handleSelectChunkAsCitation}
          />
        </section>
      </main>
    </div>
  );
}
