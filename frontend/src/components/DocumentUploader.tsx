"use client";

import React, { useRef, useState } from "react";
import {
  CheckCircle2,
  CopyCheck,
  FileUp,
  Filter,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { DocumentSummary, IngestResponse } from "@/lib/types";
import { uploadDocument } from "@/lib/api";

interface DocumentUploaderProps {
  documents: DocumentSummary[];
  selectedFilter: string | null;
  onSelectFilter: (sourceFile: string | null) => void;
  onIngestSuccess: (result: IngestResponse) => void;
}

export default function DocumentUploader({
  documents,
  selectedFilter,
  onSelectFilter,
  onIngestSuccess,
}: DocumentUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [lastResult, setLastResult] = useState<IngestResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    setLastResult(null);

    try {
      const response = await uploadDocument(file);
      setLastResult(response);
      onIngestSuccess(response);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <div className="border-b border-slate-800/80 bg-slate-900/50 px-5 py-3.5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.md,.markdown,.txt"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            disabled={isUploading}
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isUploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Indexing Hybrid Vectors...
              </>
            ) : (
              <>
                <FileUp className="h-4 w-4" />
                Ingest Document (PDF / MD / TXT)
              </>
            )}
          </button>

          <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-1.5 text-xs">
            <Filter className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-400">Scope:</span>
            <select
              value={selectedFilter || ""}
              onChange={(e) => onSelectFilter(e.target.value || null)}
              className="bg-transparent font-medium text-slate-200 focus:outline-none"
            >
              <option value="" className="bg-slate-900">
                All Indexed Documents ({documents.length})
              </option>
              {documents.map((doc) => (
                <option
                  key={doc.file_hash}
                  value={doc.source_file}
                  className="bg-slate-900"
                >
                  {doc.source_file} ({doc.chunk_count} chunks · {doc.total_pages}p)
                </option>
              ))}
            </select>
          </div>
        </div>

        {lastResult && (
          <div
            className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs ${
              lastResult.status === "ingested"
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}
          >
            {lastResult.status === "ingested" ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            ) : (
              <CopyCheck className="h-3.5 w-3.5 shrink-0" />
            )}
            <span>
              <strong>{lastResult.source_file}</strong>:{" "}
              {lastResult.status === "ingested"
                ? `${lastResult.chunks_indexed} hybrid chunks indexed (${lastResult.processing_time_ms.toFixed(0)}ms)`
                : `Idempotent skip — SHA-256 hash already indexed (${lastResult.chunks_indexed} chunks)`}
            </span>
          </div>
        )}

        {uploadError && (
          <div className="flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-300">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            <span>{uploadError}</span>
          </div>
        )}
      </div>
    </div>
  );
}
