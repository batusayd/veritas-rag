"use client";

import React from "react";
import {
  BookOpenCheck,
  FileSearch,
  Hash,
  Layers,
  Quote,
  Sparkles,
} from "lucide-react";
import { Citation, RetrievedChunkPreview } from "@/lib/types";

interface SourceViewerPanelProps {
  selectedCitation: Citation | null;
  retrievedChunks: RetrievedChunkPreview[];
  onSelectChunkAsCitation: (chunk: RetrievedChunkPreview) => void;
}

export default function SourceViewerPanel({
  selectedCitation,
  retrievedChunks,
  onSelectChunkAsCitation,
}: SourceViewerPanelProps) {
  const activeChunk =
    retrievedChunks.find(
      (c) =>
        selectedCitation &&
        c.source_file === selectedCitation.source_file &&
        c.chunk_index === selectedCitation.chunk_index
    ) || retrievedChunks[0];

  const renderHighlightedText = (fullText: string, quote?: string) => {
    if (!quote || !quote.trim()) {
      return <span>{fullText}</span>;
    }

    const cleanQuote = quote.trim();
    const idx = fullText.toLowerCase().indexOf(cleanQuote.toLowerCase());
    if (idx === -1) {
      return <span>{fullText}</span>;
    }

    const before = fullText.slice(0, idx);
    const match = fullText.slice(idx, idx + cleanQuote.length);
    const after = fullText.slice(idx + cleanQuote.length);

    return (
      <>
        <span>{before}</span>
        <mark className="rounded bg-amber-400/25 px-1 py-0.5 font-medium text-amber-200 ring-1 ring-amber-400/50">
          {match}
        </mark>
        <span>{after}</span>
      </>
    );
  };

  return (
    <div className="flex h-full flex-col border-l border-slate-800/80 bg-slate-900/30">
      {/* Right Pane Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 bg-slate-900/60 px-5 py-3.5">
        <div className="flex items-center gap-2">
          <BookOpenCheck className="h-4 w-4 text-indigo-400" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Source &amp; Citation Inspector
          </h2>
        </div>
        {retrievedChunks.length > 0 && (
          <span className="rounded-full border border-slate-700 bg-slate-800 px-2.5 py-0.5 font-mono text-[11px] text-slate-300">
            {retrievedChunks.length} Reranked Chunks
          </span>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 space-y-5 overflow-y-auto p-5">
        {!selectedCitation && !activeChunk ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 text-slate-500">
              <FileSearch className="h-6 w-6" />
            </div>
            <p className="text-sm font-medium text-slate-300">
              No Source Citation Selected
            </p>
            <p className="mt-1 max-w-xs text-xs text-slate-500">
              Click any citation card in the chat response on the left to inspect its
              source document, page number, RRF + FlashRank scores, and highlighted quote.
            </p>
          </div>
        ) : (
          <>
            {/* Active Citation Banner */}
            {selectedCitation && (
              <div className="rounded-2xl border border-indigo-500/40 bg-indigo-500/10 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-indigo-300">
                    Active Verbatim Citation
                  </span>
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="rounded bg-slate-900/90 px-2 py-0.5 text-slate-200">
                      {selectedCitation.source_file}
                    </span>
                    <span className="rounded bg-indigo-500/20 px-2 py-0.5 text-indigo-200">
                      Page {selectedCitation.page_number}
                    </span>
                    <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-emerald-300">
                      Score: {(selectedCitation.relevance_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="mt-3 flex items-start gap-2 rounded-xl bg-slate-950/80 p-3 text-xs leading-relaxed text-amber-200">
                  <Quote className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                  <p className="italic">&ldquo;{selectedCitation.exact_quote}&rdquo;</p>
                </div>
              </div>
            )}

            {/* Full Chunk Context Viewer */}
            {activeChunk && (
              <div className="rounded-2xl border border-slate-800 bg-slate-950/90 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      {activeChunk.source_file}
                    </h3>
                    <p className="text-xs text-slate-400">
                      Section:{" "}
                      <span className="text-slate-200">{activeChunk.section_title}</span>
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
                    <span className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-slate-300">
                      Page {activeChunk.page_number}
                    </span>
                    <span className="rounded border border-slate-800 bg-slate-900 px-2 py-0.5 text-slate-300">
                      Chunk #{activeChunk.chunk_index}
                    </span>
                    <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-cyan-300">
                      RRF: {activeChunk.rrf_score.toFixed(3)}
                    </span>
                    <span className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                      Rerank: {activeChunk.rerank_score.toFixed(3)}
                    </span>
                  </div>
                </div>

                <div className="mt-3 whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-300">
                  {renderHighlightedText(
                    activeChunk.text,
                    selectedCitation?.exact_quote
                  )}
                </div>
              </div>
            )}

            {/* All Retrieved & Reranked Chunks Selector */}
            {retrievedChunks.length > 0 && (
              <div className="space-y-2.5">
                <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  <Layers className="h-3.5 w-3.5 text-slate-400" />
                  <span>Top-K Reranked Context Window</span>
                </div>

                <div className="space-y-2">
                  {retrievedChunks.map((chunk) => {
                    const isActive =
                      activeChunk?.point_id === chunk.point_id;
                    return (
                      <button
                        key={chunk.point_id}
                        type="button"
                        onClick={() => onSelectChunkAsCitation(chunk)}
                        className={`w-full rounded-xl border p-3 text-left transition ${
                          isActive
                            ? "border-indigo-500/60 bg-indigo-500/10"
                            : "border-slate-800/80 bg-slate-900/50 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs">
                          <span className="flex items-center gap-1.5 font-semibold text-slate-200">
                            <Hash className="h-3.5 w-3.5 text-indigo-400" />
                            {chunk.source_file} · p.{chunk.page_number} (Chunk #{chunk.chunk_index})
                          </span>
                          <span className="flex items-center gap-2 font-mono text-[11px]">
                            <span className="text-cyan-400">
                              RRF {chunk.rrf_score.toFixed(3)}
                            </span>
                            <span className="text-emerald-400">
                              <Sparkles className="mr-0.5 inline h-3 w-3" />
                              {chunk.rerank_score.toFixed(3)}
                            </span>
                          </span>
                        </div>
                        <p className="mt-1.5 line-clamp-2 text-xs text-slate-400">
                          {chunk.text}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
