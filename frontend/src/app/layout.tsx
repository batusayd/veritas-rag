import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VeritasRAG — Hybrid Document Intelligence Engine",
  description:
    "Full-Stack Hybrid RAG Engine with Qdrant Dense+BM25 RRF Search, FlashRank Reranking, and Strict Citations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="h-screen overflow-hidden bg-slate-950 text-slate-100">
        {children}
      </body>
    </html>
  );
}
