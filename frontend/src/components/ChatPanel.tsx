"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  Bot,
  CheckCircle,
  Loader2,
  Send,
  ShieldAlert,
  Sparkles,
  User,
} from "lucide-react";
import { ChatTurn, Citation, RetrievedChunkPreview } from "@/lib/types";
import CitationCard from "./CitationCard";
import PerformanceBadge from "./PerformanceBadge";

interface ChatPanelProps {
  turns: ChatTurn[];
  selectedCitation: Citation | null;
  onSelectCitation: (
    citation: Citation,
    turnChunks: RetrievedChunkPreview[]
  ) => void;
  onSendQuery: (query: string) => void;
  isSubmitting: boolean;
}

export default function ChatPanel({
  turns,
  selectedCitation,
  onSelectCitation,
  onSendQuery,
  isSubmitting,
}: ChatPanelProps) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isSubmitting) return;
    setInput("");
    onSendQuery(trimmed);
  };

  return (
    <div className="flex h-full flex-col bg-slate-950">
      {/* Message Stream */}
      <div className="flex-1 space-y-6 overflow-y-auto p-5">
        {turns.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-indigo-500/30 bg-indigo-500/10 text-indigo-400">
              <Sparkles className="h-7 w-7" />
            </div>
            <h2 className="text-lg font-semibold text-slate-100">
              Hybrid RAG Document Intelligence
            </h2>
            <p className="mt-1.5 max-w-md text-xs leading-relaxed text-slate-400">
              Upload a PDF, Markdown, or TXT document above, then ask complex technical
              questions. Every response combines{" "}
              <span className="text-slate-200">Dense Vectors + Sparse BM25</span> with{" "}
              <span className="text-slate-200">Reciprocal Rank Fusion (RRF)</span>,{" "}
              <span className="text-slate-200">FlashRank Cross-Encoder Reranking</span>, and{" "}
              <span className="text-slate-200">strict verbatim citations</span>.
            </p>
          </div>
        ) : (
          turns.map((turn) => (
            <div key={turn.id} className="space-y-4">
              {/* User message */}
              <div className="flex items-start justify-end gap-3">
                <div className="max-w-[82%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white shadow">
                  {turn.userQuery}
                </div>
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-300">
                  <User className="h-4 w-4" />
                </div>
              </div>

              {/* Assistant response */}
              <div className="flex items-start gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-slate-800 bg-slate-900 text-emerald-400">
                  <Bot className="h-4 w-4" />
                </div>

                <div className="min-w-0 flex-1 rounded-2xl rounded-tl-sm border border-slate-800/90 bg-slate-900/40 p-4">
                  {turn.isLoading && (
                    <div className="flex items-center gap-2.5 text-xs text-slate-400">
                      <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
                      <span>
                        Rewriting query → Executing Qdrant Hybrid RRF → FlashRank Cross-Encoder → Synthesizing structured citations...
                      </span>
                    </div>
                  )}

                  {turn.error && (
                    <div className="flex items-center gap-2 text-xs text-rose-400">
                      <AlertTriangle className="h-4 w-4 shrink-0" />
                      <span>{turn.error}</span>
                    </div>
                  )}

                  {turn.response && (
                    <div className="space-y-4">
                      {/* Header metadata: Rewritten query & Confidence */}
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5 text-xs">
                        <div className="flex items-center gap-1.5 text-slate-400">
                          <Sparkles className="h-3.5 w-3.5 text-violet-400" />
                          <span>Optimized Query:</span>
                          <code className="rounded bg-slate-900 px-1.5 py-0.5 text-violet-300">
                            {turn.response.rewritten_query}
                          </code>
                        </div>

                        <div className="flex items-center gap-2">
                          {turn.response.refused ? (
                            <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-amber-300">
                              <ShieldAlert className="h-3 w-3" />
                              Hallucination Guardrail: Refused
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300">
                              <CheckCircle className="h-3 w-3" />
                              Confidence: {Math.round(turn.response.confidence_score * 100)}%
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Synthesized Answer */}
                      <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
                        {turn.response.answer}
                      </div>

                      {/* Interactive Source Citations */}
                      {turn.response.citations.length > 0 && (
                        <div className="space-y-2 pt-1">
                          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                            Verified Source Citations ({turn.response.citations.length}) — Click to Inspect in Right Pane
                          </p>
                          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                            {turn.response.citations.map((cit, idx) => (
                              <CitationCard
                                key={`${cit.source_file}-${cit.chunk_index}-${idx}`}
                                citation={cit}
                                index={idx}
                                isSelected={
                                  selectedCitation?.source_file === cit.source_file &&
                                  selectedCitation?.chunk_index === cit.chunk_index &&
                                  selectedCitation?.exact_quote === cit.exact_quote
                                }
                                onSelect={(c) =>
                                  onSelectCitation(c, turn.response!.retrieved_chunks)
                                }
                              />
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Performance Breakdown Badge */}
                      <PerformanceBadge latency={turn.response.latency} />

                      {/* Suggested Follow-up Questions */}
                      {turn.response.follow_up_questions.length > 0 && (
                        <div className="pt-1">
                          <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-500">
                            Suggested Follow-Up Queries
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {turn.response.follow_up_questions.map((followUp, i) => (
                              <button
                                key={i}
                                type="button"
                                disabled={isSubmitting}
                                onClick={() => onSendQuery(followUp)}
                                className="inline-flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-900/90 px-2.5 py-1.5 text-left text-xs text-slate-300 transition hover:border-indigo-500/50 hover:bg-indigo-500/10 hover:text-indigo-200"
                              >
                                <span>{followUp}</span>
                                <ArrowUpRight className="h-3 w-3 shrink-0 opacity-70" />
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Query Input Bar */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-800/80 bg-slate-900/50 p-4"
      >
        <div className="flex items-center gap-2 rounded-xl border border-slate-700/80 bg-slate-950 px-3.5 py-2 focus-within:border-indigo-500">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question grounded in your indexed documents..."
            disabled={isSubmitting}
            className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!input.trim() || isSubmitting}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send className="h-3.5 w-3.5" />
            <span>Ask</span>
          </button>
        </div>
      </form>
    </div>
  );
}
