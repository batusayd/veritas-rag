"use client";

import React from "react";
import { Activity, Cpu, Database, Sparkles, Timer, Zap } from "lucide-react";
import { LatencyBreakdown } from "@/lib/types";

interface PerformanceBadgeProps {
  latency: LatencyBreakdown;
}

export default function PerformanceBadge({ latency }: PerformanceBadgeProps) {
  const stages = [
    {
      label: "Rewrite",
      ms: latency.query_rewrite_ms,
      icon: Sparkles,
      color: "text-violet-400 border-violet-500/30 bg-violet-500/10",
    },
    {
      label: "Embed (Dense+BM25)",
      ms: latency.embedding_ms,
      icon: Cpu,
      color: "text-cyan-400 border-cyan-500/30 bg-cyan-500/10",
    },
    {
      label: "Qdrant RRF",
      ms: latency.qdrant_search_ms,
      icon: Database,
      color: "text-emerald-400 border-emerald-500/30 bg-emerald-500/10",
    },
    {
      label: "FlashRank",
      ms: latency.rerank_ms,
      icon: Zap,
      color: "text-amber-400 border-amber-500/30 bg-amber-500/10",
    },
    {
      label: "LLM Synthesis",
      ms: latency.llm_generation_ms,
      icon: Activity,
      color: "text-sky-400 border-sky-500/30 bg-sky-500/10",
    },
  ];

  return (
    <div className="mt-3 rounded-xl border border-slate-800/80 bg-slate-900/60 p-3">
      <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
        <span className="flex items-center gap-1.5 font-medium text-slate-300">
          <Timer className="h-3.5 w-3.5 text-emerald-400" />
          Performance Breakdown
        </span>
        <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 font-mono text-xs font-semibold text-emerald-300">
          Total: {latency.total_ms.toFixed(1)} ms
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {stages.map((stage) => {
          const Icon = stage.icon;
          return (
            <div
              key={stage.label}
              className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs ${stage.color}`}
            >
              <Icon className="h-3 w-3 shrink-0" />
              <span className="opacity-85">{stage.label}:</span>
              <span className="font-mono font-semibold">{stage.ms.toFixed(1)}ms</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
