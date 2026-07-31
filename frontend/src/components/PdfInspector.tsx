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
  downloadRunMarkdownZip,
  apiPathSegment,
  apiUrl,
} from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";

interface RunItem {
  run_name?: string;
  display_name?: string;
  files?: string[];
  has_pdf?: boolean;
}

interface TableCell {
  header: boolean;
  text: string;
  colSpan?: number;
  rowSpan?: number;
}

const decodeHtmlEntities = (value: string): string =>
  value.replace(
    /&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos);/gi,
    (entity, code: string) => {
      const normalized = code.toLowerCase();
      if (normalized === "amp") return "&";
      if (normalized === "lt") return "<";
      if (normalized === "gt") return ">";
      if (normalized === "quot") return '"';
      if (normalized === "apos") return "'";
      const radix = normalized.startsWith("#x") ? 16 : 10;
      const digits = normalized.slice(radix === 16 ? 2 : 1);
      const point = Number.parseInt(digits, radix);
      return Number.isFinite(point) && point >= 0 && point <= 0x10ffff
        ? String.fromCodePoint(point)
        : entity;
    },
  );

const parseCellSpan = (attributes: string, name: "colspan" | "rowspan") => {
  const match = attributes.match(
    new RegExp(`\\b${name}\\s*=\\s*(?:["'](\\d+)["']|(\\d+))`, "i"),
  );
  if (!match) return undefined;
  const value = Number.parseInt(match[1] || match[2], 10);
  return value >= 1 && value <= 100 ? value : undefined;
};

const parseStrictHtmlTable = (source: string): TableCell[][] | null => {
  const trimmed = source.trim();
  if (!/^<table>\s*[\s\S]*\s*<\/table>$/i.test(trimmed)) return null;

  const tags = trimmed.match(/<[^>]*>/g) || [];
  const allowedTag =
    /^<\/?(?:table|thead|tbody|tfoot|tr)>$/i;
  const allowedCell =
    /^<(?:th|td)(?:\s+(?:colspan|rowspan)\s*=\s*(?:["']\d+["']|\d+))*\s*>$|^<\/(?:th|td)>$/i;
  const allowedBreak = /^<br\s*\/?>$/i;
  if (
    tags.some(
      (tag) =>
        !allowedTag.test(tag) &&
        !allowedCell.test(tag) &&
        !allowedBreak.test(tag),
    )
  ) {
    return null;
  }

  const inner = trimmed.replace(/^<table>/i, "").replace(/<\/table>$/i, "");
  const rows: TableCell[][] = [];
  let outsideRows = "";
  let lastRowEnd = 0;
  const rowPattern = /<tr>([\s\S]*?)<\/tr>/gi;

  for (const rowMatch of inner.matchAll(rowPattern)) {
    const rowIndex = rowMatch.index ?? 0;
    outsideRows += inner.slice(lastRowEnd, rowIndex);
    lastRowEnd = rowIndex + rowMatch[0].length;

    const rowSource = rowMatch[1];
    const cells: TableCell[] = [];
    let outsideCells = "";
    let lastCellEnd = 0;
    const cellPattern = /<(th|td)((?:\s+(?:colspan|rowspan)\s*=\s*(?:["']\d+["']|\d+))*)\s*>([\s\S]*?)<\/\1>/gi;

    for (const cellMatch of rowSource.matchAll(cellPattern)) {
      const cellIndex = cellMatch.index ?? 0;
      outsideCells += rowSource.slice(lastCellEnd, cellIndex);
      lastCellEnd = cellIndex + cellMatch[0].length;
      const cellSource = cellMatch[3];
      const unsafeNestedTags = (cellSource.match(/<[^>]*>/g) || []).filter(
        (tag) => !allowedBreak.test(tag),
      );
      if (unsafeNestedTags.length > 0) return null;
      const attributes = cellMatch[2] || "";
      cells.push({
        header: cellMatch[1].toLowerCase() === "th",
        text: decodeHtmlEntities(cellSource.replace(/<br\s*\/?>/gi, "\n")),
        colSpan: parseCellSpan(attributes, "colspan"),
        rowSpan: parseCellSpan(attributes, "rowspan"),
      });
    }
    outsideCells += rowSource.slice(lastCellEnd);
    if (outsideCells.trim() || cells.length === 0) return null;
    rows.push(cells);
  }

  outsideRows += inner.slice(lastRowEnd);
  const structuralRemainder = outsideRows.replace(
    /<\/?(?:thead|tbody|tfoot)>/gi,
    "",
  );
  return rows.length > 0 && !structuralRemainder.trim() ? rows : null;
};

const renderFormattedText = (text: string) => {
  // Simple regex parser for bold, italic, and inline code
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*|`.*?`)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={index} className="font-bold text-cyan-200">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={index} className="italic text-indigo-200">{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={index} className="bg-slate-900 text-amber-300 font-mono px-1 py-0.5 rounded text-[11px] border border-slate-800">{part.slice(1, -1)}</code>;
    }
    return part;
  });
};

const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
  if (!content) {
    return <div className="text-slate-500 italic p-4">No markdown content loaded.</div>;
  }

  // OCR output may contain basic HTML tables. Only a tiny structural subset is
  // accepted; anything with HTML/SVG elements or event attributes stays text.
  const parts = content.split(/(<table>[\s\S]*?<\/table>)/gi);

  return (
    <div className="prose prose-invert max-w-none text-xs space-y-3">
      {parts.map((part, idx) => {
        if (part.toLowerCase().startsWith("<table")) {
          const rows = parseStrictHtmlTable(part);
          if (rows) {
            return (
              <div
                key={idx}
                className="my-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/90 shadow-md p-1"
              >
                <table className="w-full text-xs border-collapse">
                  <tbody>
                    {rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => {
                          const Cell = cell.header ? "th" : "td";
                          return (
                            <Cell
                              key={cellIndex}
                              colSpan={cell.colSpan}
                              rowSpan={cell.rowSpan}
                              className={
                                cell.header
                                  ? "bg-indigo-950/80 border border-slate-800 p-2 text-left font-bold text-indigo-200"
                                  : "border border-slate-800 p-2 text-slate-300"
                              }
                            >
                              {renderFormattedText(cell.text)}
                            </Cell>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }

          return (
            <p key={idx} className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
              {part}
            </p>
          );
        }

        const lines = part.split("\n");
        return (
          <div key={idx} className="space-y-1.5">
            {lines.map((line, lIdx) => {
              const trimmed = line.trim();
              if (!trimmed) return null;

              if (trimmed === "---" || trimmed === "***") {
                return <hr key={lIdx} className="border-slate-800 my-2" />;
              }

              if (trimmed.startsWith("# ")) {
                return (
                  <h1 key={lIdx} className="text-sm font-bold text-cyan-300 border-b border-cyan-500/20 pb-1 mt-3 mb-1">
                    {renderFormattedText(trimmed.slice(2))}
                  </h1>
                );
              }
              if (trimmed.startsWith("## ")) {
                return (
                  <h2 key={lIdx} className="text-xs font-bold text-indigo-300 border-b border-indigo-500/20 pb-1 mt-2 mb-1">
                    {renderFormattedText(trimmed.slice(3))}
                  </h2>
                );
              }
              if (trimmed.startsWith("### ")) {
                return (
                  <h3 key={lIdx} className="text-xs font-semibold text-slate-200 mt-2 mb-1">
                    {renderFormattedText(trimmed.slice(4))}
                  </h3>
                );
              }
              if (trimmed.startsWith("#### ")) {
                return (
                  <h4 key={lIdx} className="text-xs font-semibold text-slate-300 mt-1 mb-1">
                    {renderFormattedText(trimmed.slice(5))}
                  </h4>
                );
              }
              if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                return (
                  <li key={lIdx} className="text-xs text-slate-300 ml-4 list-disc">
                    {renderFormattedText(trimmed.slice(2))}
                  </li>
                );
              }
              if (/^\d+\.\s/.test(trimmed)) {
                const match = trimmed.match(/^(\d+\.\s)(.*)/);
                return (
                  <li key={lIdx} className="text-xs text-slate-300 ml-4 list-decimal">
                    {renderFormattedText(match ? match[2] : trimmed)}
                  </li>
                );
              }
              if (trimmed.startsWith("> ")) {
                return (
                  <blockquote key={lIdx} className="border-l-2 border-indigo-500 pl-3 py-1 bg-indigo-950/30 text-indigo-200 rounded-r-lg text-xs italic">
                    {renderFormattedText(trimmed.slice(2))}
                  </blockquote>
                );
              }
              return (
                <p key={lIdx} className="text-xs text-slate-300 leading-relaxed">
                  {renderFormattedText(line)}
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
  const runChangeRequestRef = useRef<number>(0);

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

    const handleCasesUpdated = () => {
      if (isMounted) {
        init();
      }
    };

    if (typeof window !== "undefined") {
      window.addEventListener("casesUpdated", handleCasesUpdated);
    }

    return () => {
      isMounted = false;
      if (typeof window !== "undefined") {
        window.removeEventListener("casesUpdated", handleCasesUpdated);
      }
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
    const requestId = ++runChangeRequestRef.current;
    setSelectedRun(runName);
    setSelectedFilename("");
    setRunFiles([]);
    setRawMarkdown("Loading selected document...");
    setPagesMarkdown({});
    setCurrentPage(1);
    setTotalPages(1);
    const files = await fetchRunFiles(runName);
    if (requestId !== runChangeRequestRef.current) return;
    setRunFiles(files);
    setSelectedFilename(files[0] || "");
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
  const selectedRunHasPdf = Boolean(
    runs.find((run) => (run.run_name || run.display_name) === selectedRun)?.has_pdf,
  );

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
                  Source Verification Required
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
              <label htmlFor="inspector-run" className="block text-[11px] font-semibold text-slate-400 mb-1">
                📄 Select Processed Document Run
              </label>
              <select
                id="inspector-run"
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
                <label htmlFor="inspector-file" className="block text-[11px] font-semibold text-slate-400 mb-1">
                  File
                </label>
                <select
                  id="inspector-file"
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
            <button
              type="button"
              onClick={() => selectedRun && void downloadRunMarkdownZip(selectedRun)}
              disabled={!selectedRun || runFiles.length === 0}
              className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 font-semibold flex items-center gap-1.5 border border-slate-700 text-xs cursor-pointer select-none"
            >
              <Download className="w-3.5 h-3.5 pointer-events-none" /> Download All (ZIP)
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
              {selectedRun && selectedRunHasPdf ? (
                <iframe
                  key={`${selectedRun}-${selectedFilename}`}
                  src={`${apiUrl(`/api/documents/runs/${apiPathSegment(selectedRun)}/pdf`)}#page=${currentPage}`}
                  className="w-full h-full rounded-lg border-0 bg-slate-950"
                  title="Source PDF Viewer"
                  loading="eager"
                />
              ) : (
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg text-center space-y-2 my-auto">
                  <div className="w-12 h-12 mx-auto rounded-xl bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold">
                    PDF
                  </div>
                  <div className="text-xs font-semibold text-slate-200">
                    {selectedRun
                      ? "No original PDF is attached to this Markdown run."
                      : selectedFilename || "Select a processed document."}
                  </div>
                  {selectedRun && (
                    <div className="text-[11px] text-amber-300 font-mono">
                      Markdown only; original-PDF page provenance is unavailable.
                    </div>
                  )}
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
