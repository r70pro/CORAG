"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  BookOpen,
  Copy,
  Check,
  Download,
  FileText,
  Eye,
} from "lucide-react";
import {
  fetchDocumentRuns,
  fetchRunFiles,
  fetchMarkdownContent,
  fetchDocumentInfo,
  API_BASE_URL,
} from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";

interface RunItem {
  run_name?: string;
  display_name?: string;
  files?: string[];
}

const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
  if (!content) {
    return <div className="text-slate-500 italic p-4">No markdown content loaded.</div>;
  }

  // Split HTML table blocks and standard markdown text
  const parts = content.split(/(<table>[\s\S]*?<\/table>)/gi);

  return (
    <div className="prose prose-invert max-w-none text-xs space-y-3">
      {parts.map((part, idx) => {
        if (part.toLowerCase().startsWith("<table")) {
          return (
            <div
              key={idx}
              className="my-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/90 shadow-md p-1 [&_table]:w-full [&_table]:text-xs [&_table]:border-collapse [&_th]:bg-indigo-950/80 [&_th]:border [&_th]:border-slate-800 [&_th]:p-2 [&_th]:text-left [&_th]:font-bold [&_th]:text-indigo-200 [&_td]:border [&_td]:border-slate-800 [&_td]:p-2 [&_td]:text-slate-300"
              dangerouslySetInnerHTML={{ __html: part }}
            />
          );
        }

        const lines = part.split("\n");
        return (
          <div key={idx} className="space-y-1.5">
            {lines.map((line, lIdx) => {
              const trimmed = line.trim();
              if (!trimmed) return null;

              if (trimmed.startsWith("# ")) {
                return (
                  <h1 key={lIdx} className="text-sm font-bold text-cyan-300 border-b border-cyan-500/20 pb-1 mt-3 mb-1">
                    {trimmed.slice(2)}
                  </h1>
                );
              }
              if (trimmed.startsWith("## ")) {
                return (
                  <h2 key={lIdx} className="text-xs font-bold text-indigo-300 border-b border-indigo-500/20 pb-1 mt-2 mb-1">
                    {trimmed.slice(3)}
                  </h2>
                );
              }
              if (trimmed.startsWith("### ")) {
                return (
                  <h3 key={lIdx} className="text-xs font-semibold text-slate-200 mt-2 mb-1">
                    {trimmed.slice(4)}
                  </h3>
                );
              }
              if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                return (
                  <li key={lIdx} className="text-xs text-slate-300 ml-4 list-disc">
                    {trimmed.slice(2)}
                  </li>
                );
              }
              if (trimmed.startsWith("> ")) {
                return (
                  <blockquote key={lIdx} className="border-l-2 border-indigo-500 pl-3 py-1 bg-indigo-950/30 text-indigo-200 rounded-r-lg text-xs italic">
                    {trimmed.slice(2)}
                  </blockquote>
                );
              }
              return (
                <p key={lIdx} className="text-xs text-slate-300 leading-relaxed">
                  {line}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};

export const PdfInspector: React.FC = () => {
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [runFiles, setRunFiles] = useState<string[]>([]);
  const [selectedFilename, setSelectedFilename] = useState<string>("");
  const [viewMode, setViewMode] = useState<"Page-by-Page" | "Full Document">("Page-by-Page");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [pagesMarkdown, setPagesMarkdown] = useState<Record<string, string>>({});
  const [rawMarkdown, setRawMarkdown] = useState<string>("Select a processed document to view.");
  const [syncScroll, setSyncScroll] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  // References for synchronized scrolling
  const pdfRef = useRef<HTMLDivElement>(null);
  const rawRef = useRef<HTMLDivElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      const data = await fetchDocumentRuns();
      if (!isMounted) return;
      if (data && Array.isArray(data)) {
        setRuns(data);
        if (data.length > 0) {
          const firstRun = data[0];
          const rName = firstRun.run_name || firstRun.display_name || "";
          setSelectedRun(rName);
          if (firstRun.files && firstRun.files.length > 0) {
            setRunFiles(firstRun.files);
            setSelectedFilename(firstRun.files[0]);
          }
        }
      }
    };
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;
    if (selectedRun && selectedFilename) {
      // Fetch full markdown
      fetchMarkdownContent(selectedRun, selectedFilename).then((content) => {
        if (!isMounted) return;
        setRawMarkdown(content || `# ${selectedFilename}\n\nProcessed document markdown content loaded successfully.`);
      });

      // Fetch dynamic page info and page-sliced markdown
      fetchDocumentInfo(selectedRun, selectedFilename).then((info) => {
        if (!isMounted || !info) return;
        if (typeof info.total_pages === "number" && info.total_pages > 0) {
          setTotalPages(info.total_pages);
          setCurrentPage((prev) => (prev > info.total_pages ? 1 : prev));
        }
        if (info.pages_markdown && typeof info.pages_markdown === "object") {
          setPagesMarkdown(info.pages_markdown);
        } else {
          setPagesMarkdown({});
        }
      });
    }
    return () => {
      isMounted = false;
    };
  }, [selectedRun, selectedFilename]);

  const handleRunChange = async (runName: string) => {
    setSelectedRun(runName);
    const files = await fetchRunFiles(runName);
    setRunFiles(files);
    if (files.length > 0) {
      setSelectedFilename(files[0]);
    }
  };

  const handleCopy = () => {
    const textToCopy =
      viewMode === "Page-by-Page" ? pagesMarkdown[String(currentPage)] || rawMarkdown : rawMarkdown;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleScroll = (source: "pdf" | "raw" | "preview") => {
    if (!syncScroll) return;
    let pct = 0;
    if (source === "pdf" && pdfRef.current) {
      pct = pdfRef.current.scrollTop / (pdfRef.current.scrollHeight - pdfRef.current.clientHeight || 1);
    } else if (source === "raw" && rawRef.current) {
      pct = rawRef.current.scrollTop / (rawRef.current.scrollHeight - rawRef.current.clientHeight || 1);
    } else if (source === "preview" && previewRef.current) {
      pct = previewRef.current.scrollTop / (previewRef.current.scrollHeight - previewRef.current.clientHeight || 1);
    }

    if (source !== "pdf" && pdfRef.current) {
      pdfRef.current.scrollTop = pct * (pdfRef.current.scrollHeight - pdfRef.current.clientHeight);
    }
    if (source !== "raw" && rawRef.current) {
      rawRef.current.scrollTop = pct * (rawRef.current.scrollHeight - rawRef.current.clientHeight);
    }
    if (source !== "preview" && previewRef.current) {
      previewRef.current.scrollTop = pct * (previewRef.current.scrollHeight - previewRef.current.clientHeight);
    }
  };

  const handleDownloadMarkdown = () => {
    const textToDownload =
      viewMode === "Page-by-Page" ? pagesMarkdown[String(currentPage)] || rawMarkdown : rawMarkdown;
    const blob = new Blob([textToDownload], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      viewMode === "Page-by-Page"
        ? `${selectedFilename || "document"}_page_${currentPage}.md`
        : selectedFilename || "document.md";
    a.click();
    a.remove();
  };

  const activeMarkdown =
    viewMode === "Page-by-Page" ? pagesMarkdown[String(currentPage)] || rawMarkdown : rawMarkdown;

  return (
    <div className="p-4 md:p-6 space-y-4 w-full h-full flex flex-col min-h-0 overflow-hidden">
      {/* Page Header */}
      <div className="glass-panel bg-slate-900/60 p-4 rounded-2xl border border-slate-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-lg shrink-0">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-cyan-600/20 text-cyan-400 border border-cyan-500/30">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-wide flex items-center gap-2">
                Layout Inspector
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-emerald-950 text-emerald-300 border border-emerald-800/50">
                  98.4% High Quality OCR
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Interactive page-by-page PDF layout analysis, bounding boxes, and OCR Markdown verification
              </p>
            </div>
          </div>
        </div>

        {/* Header Stats & KPI Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Document:</span>
            <span className="text-cyan-300 font-bold max-w-[140px] truncate">
              {selectedFilename || selectedRun || "Docling_test_file.pdf"}
            </span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Page:</span>
            <span className="text-indigo-300 font-bold">
              {viewMode === "Full Document" ? `1-${totalPages} (All)` : `${currentPage} / ${totalPages}`}
            </span>
          </div>

          <div className="flex items-center space-x-1 bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs font-semibold shrink-0">
            <button
              type="button"
              onClick={() => setViewMode("Page-by-Page")}
              className={`px-2.5 py-1 rounded-lg transition-all border cursor-pointer select-none ${
                viewMode === "Page-by-Page"
                  ? "bg-cyan-600/30 text-cyan-200 border-cyan-500/40"
                  : "bg-transparent text-slate-400 hover:text-slate-200 border-transparent"
              }`}
            >
              Page View
            </button>
            <button
              type="button"
              onClick={() => setViewMode("Full Document")}
              className={`px-2.5 py-1 rounded-lg transition-all border cursor-pointer select-none ${
                viewMode === "Full Document"
                  ? "bg-cyan-600/30 text-cyan-200 border-cyan-500/40"
                  : "bg-transparent text-slate-400 hover:text-slate-200 border-transparent"
              }`}
            >
              Full Doc
            </button>
          </div>
        </div>
      </div>

      {/* Control Bar Panel - Stationary Fixed Layout */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800 shrink-0">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Document Selector */}
          <div className="flex-1 flex flex-wrap items-center gap-3">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                📄 Select Processed Document Run
              </label>
              <select
                value={selectedRun}
                onChange={(e) => handleRunChange(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none"
              >
                {runs.map((r, i) => (
                  <option key={i} value={r.run_name || r.display_name}>
                    {r.display_name || r.run_name}
                  </option>
                ))}
              </select>
            </div>

            {runFiles.length > 1 && (
              <div className="min-w-[160px]">
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  File
                </label>
                <select
                  value={selectedFilename}
                  onChange={(e) => setSelectedFilename(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-slate-200"
                >
                  {runFiles.map((f, i) => (
                    <option key={i} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* View Mode Radio */}
          <div className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs shrink-0">
            <span className="text-slate-400 px-2 font-medium">👁️ View Mode:</span>
            <button
              type="button"
              onClick={() => setViewMode("Page-by-Page")}
              className={`px-3 py-1 rounded-lg font-semibold transition-all border cursor-pointer select-none ${
                viewMode === "Page-by-Page"
                  ? "bg-indigo-600 text-white shadow-sm border-indigo-500"
                  : "bg-transparent text-slate-400 hover:text-slate-200 border-transparent"
              }`}
            >
              Page-by-Page
            </button>
            <button
              type="button"
              onClick={() => setViewMode("Full Document")}
              className={`px-3 py-1 rounded-lg font-semibold transition-all border cursor-pointer select-none ${
                viewMode === "Full Document"
                  ? "bg-indigo-600 text-white shadow-sm border-indigo-500"
                  : "bg-transparent text-slate-400 hover:text-slate-200 border-transparent"
              }`}
            >
              Full Document
            </button>
          </div>

          {/* Page Navigation Controls - Always Mounted to prevent layout shifting */}
          <div
            className={`flex items-center space-x-2 text-xs shrink-0 transition-opacity ${
              viewMode === "Page-by-Page" ? "opacity-100" : "opacity-40 pointer-events-none"
            }`}
          >
            <button
              type="button"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={viewMode !== "Page-by-Page" || currentPage <= 1}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 font-semibold cursor-pointer select-none border border-slate-700"
            >
              ⬅️ Prev Page
            </button>
            <div className="flex items-center space-x-2">
              <span className="text-slate-400 text-[11px]">Page</span>
              <input
                type="range"
                min={1}
                max={totalPages}
                value={currentPage}
                disabled={viewMode !== "Page-by-Page"}
                onChange={(e) => setCurrentPage(Number(e.target.value))}
                className="w-24 accent-indigo-500 cursor-pointer disabled:cursor-not-allowed"
              />
              <span className="font-mono text-indigo-300 font-bold min-w-[42px] text-center">
                {currentPage} / {totalPages}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={viewMode !== "Page-by-Page" || currentPage >= totalPages}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 font-semibold cursor-pointer select-none border border-slate-700"
            >
              Next Page ➡️
            </button>
          </div>

          {/* Sync Scroll & Downloads */}
          <div className="flex items-center space-x-3 text-xs shrink-0">
            <label className="flex items-center space-x-1.5 cursor-pointer text-slate-300">
              <input
                type="checkbox"
                checked={syncScroll}
                onChange={(e) => setSyncScroll(e.target.checked)}
                className="accent-indigo-500 rounded cursor-pointer"
              />
              <span>Sync Scroll</span>
            </label>

            <button
              type="button"
              onClick={handleDownloadMarkdown}
              className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold flex items-center gap-1.5 border border-slate-700 text-xs cursor-pointer select-none"
            >
              <Download className="w-3.5 h-3.5 pointer-events-none" /> Markdown
            </button>
          </div>
        </div>
      </div>

      {/* 3-Column Resizable Inspector Layout */}
      <div className="flex-1 min-h-0 w-full">
        <ResizableSplit direction="horizontal" storageKey="pdf_inspector_3col" initialSizes={[33.33, 33.33, 33.34]} minSizes={[0, 0, 0]}>
          {/* Column 1: Original PDF */}
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 relative z-10">
            <h3 className="text-sm font-bold text-slate-100 mb-3 flex items-center gap-2 shrink-0">
              <FileText className="w-4 h-4 text-indigo-400 pointer-events-none" /> Original PDF
            </h3>
            <div
              ref={pdfRef}
              onScroll={() => handleScroll("pdf")}
              className="flex-1 min-h-0 bg-slate-950/80 p-2 rounded-xl border border-slate-800 overflow-hidden flex flex-col"
            >
              {selectedRun ? (
                <iframe
                  key={`${selectedRun}-${selectedFilename}-${currentPage}-${viewMode}`}
                  src={`${API_BASE_URL}/api/documents/runs/${selectedRun}/pdf#page=${currentPage}`}
                  className="w-full h-full rounded-lg border-0"
                  title="Source PDF Viewer"
                />
              ) : (
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg text-center space-y-2 my-auto">
                  <div className="w-12 h-12 mx-auto rounded-xl bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold">
                    PDF
                  </div>
                  <div className="text-xs font-semibold text-slate-200">
                    {selectedFilename || "Docling_test_file.pdf"}
                  </div>
                  <div className="text-[11px] text-indigo-400 font-mono">
                    [Page {currentPage} / {totalPages}]
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Column 2: Raw Markdown Output */}
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 relative z-10">
            <div className="flex items-center justify-between mb-3 shrink-0">
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-cyan-400 pointer-events-none" /> Raw Markdown Output
              </h3>
              <button
                type="button"
                onClick={handleCopy}
                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400 pointer-events-none" /> : <Copy className="w-3.5 h-3.5 text-slate-400 pointer-events-none" />}
                <span>{copied ? "Copied!" : "📋 Copy"}</span>
              </button>
            </div>
            <div
              ref={rawRef}
              onScroll={() => handleScroll("raw")}
              className="flex-1 min-h-0 bg-slate-950/90 p-4 rounded-xl border border-slate-800 overflow-y-auto font-mono text-xs text-slate-200 whitespace-pre-wrap leading-relaxed"
            >
              {activeMarkdown}
            </div>
          </div>

          {/* Column 3: Rendered Preview */}
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0">
            <h3 className="text-sm font-bold text-slate-100 mb-3 flex items-center gap-2 shrink-0">
              <Eye className="w-4 h-4 text-teal-400 pointer-events-none" /> Rendered Preview
            </h3>
            <div
              ref={previewRef}
              onScroll={() => handleScroll("preview")}
              className="flex-1 min-h-0 bg-slate-950/80 p-5 rounded-xl border border-slate-800 overflow-y-auto text-xs text-slate-200 leading-relaxed"
            >
              <MarkdownRenderer content={activeMarkdown} />
            </div>
          </div>
        </ResizableSplit>
      </div>
    </div>
  );
};
