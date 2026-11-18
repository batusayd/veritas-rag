"use client";

import React from "react";
import { BookOpen, FileText, Quote } from "lucide-react";
import { Citation } from "@/lib/types";

interface CitationCardProps {
  citation: Citation;
  index: number;
  isSelected: boolean;
  onSelect: (citation: Citation) => void;
}

export default function CitationCard({
  citation,
  index,
  isSelected,
  onSelect,
}: CitationCardProps) {
  const scorePct = Math.round(citation.relevance_score * 100);

  return (
    <button
      type="button"
      onClick={() => onSelect(citation)}
      className={`group w-full rounded-xl border p-3 text-left transition-all ${
        isSelected
          ? "border-indigo-500 bg-indigo-500/15 shadow-lg shadow-indigo-500/10"
          : "border-slate-800 bg-slate-900/70 hover:border-slate-700 hover:bg-slate-800/70"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 overflow-hidden">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-indigo-500/20 font-mono text-xs font-bold text-indigo-300">
            [{index + 1}]
          </span>
          <FileText className="h-3.5 w-3.5 shrink-0 text-indigo-400" />
          <span className="truncate text-xs font-semibold text-slate-200">
            {citation.source_file}
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] text-slate-300">
            p.{citation.page_number}
          </span>
          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[11px] text-emerald-300">
            {scorePct}% match
          </span>
        </div>
      </div>

      <div className="mt-1.5 flex items-center gap-1 text-[11px] text-slate-400">
        <BookOpen className="h-3 w-3 shrink-0 text-slate-500" />
        <span className="truncate">
          Section: <strong className="text-slate-300">{citation.section_title}</strong> · Chunk #{citation.chunk_index}
        </span>
      </div>

      <div className="mt-2 flex items-start gap-1.5 rounded-lg bg-slate-950/70 p-2 text-xs text-slate-300">
        <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-indigo-400/80" />
        <p className="line-clamp-2 italic text-slate-300/90">
          &ldquo;{citation.exact_quote}&rdquo;
        </p>
      </div>
    </button>
  );
}
