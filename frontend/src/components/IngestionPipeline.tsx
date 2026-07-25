"use client";

import React, { useState, useEffect } from "react";
import {
  FileSpreadsheet,
  Upload,
  Play,
  Square,
  Settings2,
  Save,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  FileText,
} from "lucide-react";
import { triggerIngestSSE, stopPipelineRun, updateSettings, fetchSettings, uploadPipelineFiles } from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";
import { ResizableBlock } from "@/components/ResizableBlock";

function parseFileStatusHtml(htmlStr: string): { name: string; pages: number | string; status: string }[] {
  if (!htmlStr) return [];
  const items: { name: string; pages: number | string; status: string }[] = [];
  const trRegex = /<tr[^>]*>\s*<td[^>]*>(.*?)<\/td>\s*<td[^>]*>(.*?)<\/td>\s*<td[^>]*>(.*?)<\/td>\s*<\/tr>/gi;
  let match;
  while ((match = trRegex.exec(htmlStr)) !== null) {
    const rawName = match[1].replace(/<[^>]+>/g, "").trim();
    const rawPages = match[2].replace(/<[^>]+>/g, "").trim();
    const rawStatus = match[3].replace(/<[^>]+>/g, "").trim();
    if (rawName && rawName !== "File") {
      items.push({ name: rawName, pages: rawPages, status: rawStatus });
    }
  }
  return items;
}

function parseManifestHtml(htmlStr: string): string[] {
  if (!htmlStr) return [];
  const items: string[] = [];
  const trRegex = /<tr[^>]*>\s*<td[^>]*>(.*?)<\/td>\s*<td[^>]*>(.*?)<\/td>\s*<td[^>]*>(.*?)<\/td>\s*<\/tr>/gi;
  let match;
  while ((match = trRegex.exec(htmlStr)) !== null) {
    const rawName = match[1].replace(/<[^>]+>/g, "").trim();
    const rawPages = match[2].replace(/<[^>]+>/g, "").trim();
    const rawSize = match[3].replace(/<[^>]+>/g, "").trim();
    if (rawName && !rawName.startsWith("Total")) {
      items.push(`${rawName} (${rawPages} pgs, ${rawSize})`);
    }
  }
  return items;
}

export const IngestionPipeline: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeRunId, setActiveRunId] = useState<string>("");
  const [statusBadge, setStatusBadge] = useState<string>("Idle");
  const [progressPct, setProgressPct] = useState<number>(0);
  const [completedPages, setCompletedPages] = useState<number>(0);
  const [failedPages, setFailedPages] = useState<number>(0);
  const [totalPages, setTotalPages] = useState<number>(9);
  const [logMessages, setLogMessages] = useState<string[]>([
    "Ready for PDF OCR ingestion.",
  ]);

  // Parameters
  const [serverUrl, setServerUrl] = useState<string>("http://localhost:8000/v1");
  const [modelName, setModelName] = useState<string>("allenai/olmOCR-2-7B-1025-FP8");
  const [workers, setWorkers] = useState<number>(4);
  const [maxConcurrent, setMaxConcurrent] = useState<number>(20);
  const [targetDim, setTargetDim] = useState<number>(1288);
  const [maxRetries, setMaxRetries] = useState<number>(8);
  const [guidedDecoding, setGuidedDecoding] = useState<boolean>(true);
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(false);
  const [configStatus, setConfigStatus] = useState<string>("");

  // Load pipeline settings from backend on mount so values stay in sync
  useEffect(() => {
    fetchSettings()
      .then((settings) => {
        if (settings?.server_url) setServerUrl(String(settings.server_url));
        if (settings?.model_name) setModelName(String(settings.model_name));
        if (settings?.workers) setWorkers(Number(settings.workers) || 4);
        if (settings?.max_concurrent_requests) setMaxConcurrent(Number(settings.max_concurrent_requests) || 20);
        if (settings?.target_longest_image_dim) setTargetDim(Number(settings.target_longest_image_dim) || 1288);
        if (settings?.max_page_retries) setMaxRetries(Number(settings.max_page_retries) || 8);
        if (settings?.guided_decoding !== undefined) setGuidedDecoding(Boolean(settings.guided_decoding));
      })
      .catch(() => {});
  }, []);

  // Source Files
  const [pdfFiles, setPdfFiles] = useState<File[]>([]);
  const [selectedFilePath, setSelectedFilePath] = useState<string>("/home/owner/Downloads/Docling_test_file.pdf");

  // Manifest & File Status HTML representations
  const [manifestItems, setManifestItems] = useState<string[]>([
    "Docling_test_file.pdf (Target PDF)",
  ]);
  const [fileStatuses, setFileStatuses] = useState<
    { name: string; pages: number | string; status: string }[]
  >([
    { name: "Docling_test_file.pdf", pages: "-", status: "Ready" },
  ]);

  const handleSaveConfig = async () => {
    setConfigStatus("Saving...");
    const res = await updateSettings({
      server_url: serverUrl,
      model_name: modelName,
      workers,
      max_concurrent: maxConcurrent,
      target_dim: targetDim,
      max_retries: maxRetries,
      guided_decoding: guidedDecoding,
    });
    setConfigStatus(res.message || "Saved");
  };

  const handleStartPipeline = async () => {
    setIsProcessing(true);
    setStatusBadge("Processing");
    setProgressPct(0);
    setCompletedPages(0);
    setFailedPages(0);

    setFileStatuses((prev) =>
      prev.map((item) => ({ ...item, status: "Processing..." }))
    );

    let targetPaths: string[] = [];

    if (pdfFiles.length > 0) {
      try {
        setLogMessages((prev) => [...prev, `[Upload] Uploading ${pdfFiles.length} file(s) to server...`]);
        targetPaths = await uploadPipelineFiles(pdfFiles);
      } catch (uploadErr) {
        setLogMessages((prev) => [...prev, `[Upload Warning] ${String(uploadErr)}. Falling back to target file path.`]);
        targetPaths = selectedFilePath ? [selectedFilePath] : [];
      }
    } else if (selectedFilePath) {
      targetPaths = [selectedFilePath];
    }

    if (targetPaths.length === 0) {
      setLogMessages((prev) => [...prev, "[Error] No valid input file selected or specified."]);
      setStatusBadge("Error");
      setIsProcessing(false);
      return;
    }

    triggerIngestSSE(
      {
        file_paths: targetPaths,
        server_url: serverUrl,
        model_name: modelName,
        workers,
        max_concurrent: maxConcurrent,
        max_retries: maxRetries,
        target_dim: targetDim,
        guided_decoding: guidedDecoding,
      },
      (eventData: unknown) => {
        const data = (eventData || {}) as Record<string, unknown>;
        if (data.error && typeof data.error === "string") {
          setLogMessages((prev) => [...prev, `[Error] ${data.error}`]);
          setStatusBadge("Failed");
          setIsProcessing(false);
          setFileStatuses((prev) =>
            prev.map((item) => ({ ...item, status: "Failed" }))
          );
          return;
        }
        if (data.log_text && typeof data.log_text === "string") {
          setLogMessages((prev) => [...prev, data.log_text as string]);
        }
        if (data.status_badge && typeof data.status_badge === "string") {
          setStatusBadge(data.status_badge);
          if (data.status_badge.includes("Failed") || data.status_badge.includes("Error") || data.status_badge.includes("Unreachable")) {
            setIsProcessing(false);
          }
        }
        if (data.run_id && typeof data.run_id === "string") {
          setActiveRunId(data.run_id);
        }

        if (typeof data.file_status_html === "string" && data.file_status_html.trim()) {
          const parsedStatus = parseFileStatusHtml(data.file_status_html);
          if (parsedStatus.length > 0) {
            setFileStatuses(parsedStatus);
          }
        }

        if (typeof data.upload_manifest_html === "string" && data.upload_manifest_html.trim()) {
          const parsedManifest = parseManifestHtml(data.upload_manifest_html);
          if (parsedManifest.length > 0) {
            setManifestItems(parsedManifest);
          }
        }

        let curCompleted = completedPages;
        let curTotal = totalPages;

        if (typeof data.completed_pages === "number") {
          curCompleted = data.completed_pages;
          setCompletedPages(data.completed_pages);
        }
        if (typeof data.failed_pages === "number") {
          setFailedPages(data.failed_pages);
        }

        if (data.progress_html && typeof data.progress_html === "string") {
          const matchPages = data.progress_html.match(/(\d+)\/(\d+)\s+Pages/);
          if (matchPages && matchPages[1] && matchPages[2]) {
            curCompleted = parseInt(matchPages[1], 10);
            curTotal = parseInt(matchPages[2], 10);
            setCompletedPages(curCompleted);
            setTotalPages(curTotal);
          }
          const matchPct = data.progress_html.match(/>(\d+)%</) || data.progress_html.match(/font-semibold[^>]*>(\d+)%/);
          if (matchPct && matchPct[1]) {
            setProgressPct(Math.min(100, parseInt(matchPct[1], 10)));
          } else if (curTotal > 0) {
            setProgressPct(Math.min(100, Math.round((curCompleted / curTotal) * 100)));
          } else {
            setProgressPct(0);
          }
        } else if (curTotal > 0) {
          setProgressPct(Math.min(100, Math.round((curCompleted / curTotal) * 100)));
        } else {
          setProgressPct(0);
        }
      },
      (err) => {
        setLogMessages((prev) => [...prev, `[Error] ${String(err)}`]);
        setStatusBadge("Error");
        setIsProcessing(false);
        setFileStatuses((prev) =>
          prev.map((item) => ({ ...item, status: "Error" }))
        );
      },
      () => {
        setStatusBadge((prev) => (prev.includes("Error") || prev.includes("Failed") ? prev : "Completed"));
        setIsProcessing(false);
        setFileStatuses((prev) =>
          prev.map((item) => ({
            ...item,
            status: item.status.includes("Failed") || item.status.includes("Error") ? item.status : "Done",
          }))
        );
        setLogMessages((prev) => [...prev, "[Complete] Pipeline batch processing finished."]);
      }
    );
  };

  const handleStopPipeline = async () => {
    if (!activeRunId) return;
    setLogMessages((prev) => [...prev, "[Stop] Sending stop signal to active run..."]);
    const res = await stopPipelineRun(activeRunId);
    setLogMessages((prev) => [...prev, `[Stop Result] ${res.message}`]);
    setStatusBadge("Stopped");
    setIsProcessing(false);
  };

  return (
    <div className="p-4 md:p-6 space-y-4 w-full h-full flex flex-col min-h-0 overflow-hidden">
      {/* Page Header */}
      <div className="glass-panel bg-slate-900/60 p-4 rounded-2xl border border-slate-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-lg shrink-0">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
              <FileSpreadsheet className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-wide flex items-center gap-2">
                Ingestion Pipeline
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-indigo-950 text-indigo-300 border border-indigo-800/50">
                  vLLM OCR Engine
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                High-throughput document OCR processing, batch queuing & guided decoding
              </p>
            </div>
          </div>
        </div>

        {/* Header Stats & KPI Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Queue:</span>
            <span className="text-cyan-300 font-bold">{fileStatuses.length} files ({totalPages} pgs)</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Workers:</span>
            <span className="text-indigo-300 font-bold">{workers} Threads</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Target Dim:</span>
            <span className="text-amber-300 font-bold">{targetDim}px</span>
          </div>

          <span
            className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${
              statusBadge === "Running" || statusBadge === "Processing"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse"
                : statusBadge === "Completed"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                : "bg-slate-800 text-slate-400 border border-slate-700"
            }`}
          >
            ● {statusBadge}
          </span>
        </div>
      </div>

      <div className="flex-1 min-h-0 w-full">
        <ResizableSplit direction="horizontal" storageKey="ingestion_main" initialSizes={[25, 75]} minSizes={[0, 0]}>
          {/* Left Column: Pipeline Settings */}
          <div className="glass-panel p-4 rounded-2xl space-y-4 border border-slate-800 h-full min-h-0 overflow-y-auto relative z-10">
            <h3 className="text-xs font-bold text-slate-200 flex items-center gap-1.5 uppercase tracking-wider">
              <Settings2 className="w-4 h-4 text-indigo-400" /> Pipeline Settings
            </h3>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">vLLM OpenAI Server URL</label>
                <input
                  type="text"
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  placeholder="http://localhost:8000/v1"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Model Name</label>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200"
                >
                  <option value="allenai/olmOCR-2-7B-1025-FP8">allenai/olmOCR-2-7B-1025-FP8</option>
                  <option value="nvidia/Phi-4-reasoning-plus-NVFP4">nvidia/Phi-4-reasoning-plus-NVFP4</option>
                  <option value="Qwen/Qwen2-VL-7B-Instruct">Qwen/Qwen2-VL-7B-Instruct</option>
                </select>
              </div>

              {/* Advanced Parameters Accordion */}
              <div className="border border-slate-800 rounded-xl overflow-hidden relative z-10">
                <button
                  type="button"
                  onClick={() => setAdvancedOpen(!advancedOpen)}
                  className="w-full px-3 py-2 bg-slate-900/60 flex items-center justify-between text-xs font-bold text-slate-300 cursor-pointer select-none"
                >
                  <span>Advanced Parameters</span>
                  {advancedOpen ? <ChevronUp className="w-3.5 h-3.5 pointer-events-none" /> : <ChevronDown className="w-3.5 h-3.5 pointer-events-none" />}
                </button>

                {advancedOpen && (
                  <div className="p-3 space-y-3 bg-slate-950/40">
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Workers</span>
                        <span className="font-mono text-indigo-300">{workers}</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={64}
                        value={workers}
                        onChange={(e) => setWorkers(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Max Concurrent Requests</span>
                        <span className="font-mono text-indigo-300">{maxConcurrent}</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={2000}
                        step={10}
                        value={maxConcurrent}
                        onChange={(e) => setMaxConcurrent(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Target Image Dim</span>
                        <span className="font-mono text-indigo-300">{targetDim}</span>
                      </div>
                      <input
                        type="range"
                        min={512}
                        max={2048}
                        step={64}
                        value={targetDim}
                        onChange={(e) => setTargetDim(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Max Page Retries</span>
                        <span className="font-mono text-indigo-300">{maxRetries}</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={20}
                        value={maxRetries}
                        onChange={(e) => setMaxRetries(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>

                    <label className="flex items-center space-x-2 text-slate-300 cursor-pointer pt-1">
                      <input
                        type="checkbox"
                        checked={guidedDecoding}
                        onChange={(e) => setGuidedDecoding(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span>Guided Decoding (YAML)</span>
                    </label>
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={handleSaveConfig}
                className="w-full py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs flex items-center justify-center gap-1.5 border border-slate-700 cursor-pointer select-none"
              >
                <Save className="w-3.5 h-3.5 pointer-events-none" /> Save Configuration
              </button>
              {configStatus && <p className="text-[11px] font-mono text-emerald-400 text-center">{configStatus}</p>}
            </div>
          </div>

          {/* Right Main Area with Vertical Resizable Split */}
          <div className="flex flex-col h-full min-h-0 pl-1">
            <ResizableSplit direction="vertical" storageKey="ingestion_log_split" initialSizes={[65, 35]} minSizes={[0, 0]}>
              {/* Top Execution, Dropzone & Manifest Area */}
              <div className="space-y-4 h-full min-h-0 overflow-y-auto pr-1">
                {/* Upload + Monitoring Row */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Upload Area */}
                  <div className="md:col-span-2 glass-panel p-4 rounded-2xl space-y-3 border border-slate-800 relative z-10">
                    <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                      <Upload className="w-4 h-4 text-indigo-400 pointer-events-none" /> Source Documents
                    </h3>

                    <div className="border-2 border-dashed border-slate-700/80 rounded-xl p-3 text-center space-y-1 hover:border-indigo-500/60 transition-colors">
                      <input
                        type="file"
                        multiple
                        accept=".pdf"
                        onChange={(e) => {
                          const files = Array.from(e.target.files || []);
                          setPdfFiles(files);
                          if (files.length > 0) {
                            setManifestItems(files.map((f) => `${f.name} (${(f.size / (1024 * 1024)).toFixed(2)} MB)`));
                            setFileStatuses(files.map((f) => ({ name: f.name, pages: "-", status: "Ready" })));
                          }
                        }}
                        className="hidden"
                        id="pdf-file-input"
                      />
                      <label htmlFor="pdf-file-input" className="cursor-pointer space-y-1 block">
                        <div className="w-8 h-8 mx-auto rounded-xl bg-indigo-600/20 text-indigo-400 flex items-center justify-center">
                          <Upload className="w-4 h-4 pointer-events-none" />
                        </div>
                        <div className="text-xs font-semibold text-slate-200">
                          {pdfFiles.length > 0
                            ? `${pdfFiles.length} file(s) selected`
                            : "Upload / Drag-and-drop PDFs"}
                        </div>
                        <div className="text-[10px] text-slate-400 font-mono">
                          Target default: {selectedFilePath}
                        </div>
                      </label>
                    </div>

                    <div>
                      <label className="block text-slate-400 text-[11px] mb-1">Target File Path / Fallback</label>
                      <input
                        type="text"
                        value={selectedFilePath}
                        onChange={(e) => {
                          setSelectedFilePath(e.target.value);
                          setManifestItems([`${e.target.value.split("/").pop()} (Target PDF)`]);
                          setFileStatuses([{ name: e.target.value.split("/").pop() || e.target.value, pages: "-", status: "Ready" }]);
                        }}
                        placeholder="/home/owner/Downloads/Docling_test_file.pdf"
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 font-mono"
                      />
                    </div>

                    <div className="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={handleStartPipeline}
                        disabled={isProcessing}
                        suppressHydrationWarning
                        className="flex-1 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/20 cursor-pointer select-none"
                      >
                        <Play className="w-4 h-4 pointer-events-none" /> Start Batch Processing
                      </button>
                      <button
                        type="button"
                        onClick={handleStopPipeline}
                        disabled={!isProcessing}
                        suppressHydrationWarning
                        className="px-4 py-2 rounded-xl bg-rose-950/60 hover:bg-rose-900/60 disabled:opacity-40 text-rose-300 font-semibold text-xs flex items-center gap-1.5 border border-rose-800/60 cursor-pointer select-none"
                      >
                        <Square className="w-3.5 h-3.5 text-rose-400 pointer-events-none" /> Stop Process
                      </button>
                    </div>
                  </div>

                  {/* Monitoring Cards */}
                  <div className="md:col-span-1 glass-panel p-4 rounded-2xl space-y-3 border border-slate-800">
                    <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                      📊 Monitoring
                    </h3>

                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                          <span>Batch Progress</span>
                          <span className="font-mono text-indigo-300">{progressPct}%</span>
                        </div>
                        <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-800">
                          <div
                            className="bg-gradient-to-r from-indigo-500 to-cyan-400 h-full transition-all duration-300 rounded-full"
                            style={{ width: `${progressPct}%` }}
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 pt-1">
                        <div className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800 text-center">
                          <div className="text-base font-bold text-emerald-400">{completedPages}</div>
                          <div className="text-[10px] text-slate-400">Completed Pages</div>
                        </div>

                        <div className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800 text-center">
                          <div className="text-base font-bold text-rose-400">{failedPages}</div>
                          <div className="text-[10px] text-slate-400">Failed Pages</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Manifest & File Status Tables (Side by Side Resizable Blocks) */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ResizableBlock
                    id="ingest_upload_manifest"
                    defaultHeight={180}
                    minHeight={100}
                    className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col min-h-0"
                    title={
                      <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5 shrink-0">
                        <FileText className="w-3.5 h-3.5 text-indigo-400" /> Upload Manifest
                      </h4>
                    }
                  >
                    <div className="flex-1 min-h-0 bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 space-y-1.5 overflow-y-auto">
                      {manifestItems.map((item, i) => (
                        <div key={i} className="flex items-center gap-2 py-0.5 border-b border-slate-900/60 last:border-0">
                          <span className="text-indigo-400 text-sm">📄</span>
                          <span className="truncate">{item}</span>
                        </div>
                      ))}
                    </div>
                  </ResizableBlock>

                  <ResizableBlock
                    id="ingest_per_file_status"
                    defaultHeight={180}
                    minHeight={100}
                    className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col min-h-0"
                    title={
                      <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5 shrink-0">
                        <Layers className="w-3.5 h-3.5 text-indigo-400" /> Per-File Status
                      </h4>
                    }
                  >
                    <div className="flex-1 min-h-0 bg-slate-950 rounded-xl border border-slate-800 overflow-y-auto text-xs">
                      <table className="w-full text-left font-mono">
                        <thead className="bg-slate-900 text-slate-400 text-[10px] sticky top-0 z-10">
                          <tr>
                            <th className="p-2">File</th>
                            <th className="p-2 text-center">Pages</th>
                            <th className="p-2 text-center">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 text-[11px] text-slate-300">
                          {fileStatuses.map((fs, i) => (
                            <tr key={i} className="hover:bg-slate-900/40 transition-colors">
                              <td className="p-2 max-w-[180px] truncate" title={fs.name}>{fs.name}</td>
                              <td className="p-2 text-center text-slate-400">{fs.pages}</td>
                              <td className="p-2 text-center">
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] font-semibold inline-block ${
                                    fs.status.includes("Done") || fs.status.includes("Success") || fs.status.includes("✓")
                                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                                      : fs.status.includes("Failed") || fs.status.includes("Error") || fs.status.includes("✗")
                                      ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                                      : fs.status.includes("Processing") || fs.status.includes("Running")
                                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse"
                                      : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                                  }`}
                                >
                                  {fs.status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </ResizableBlock>
                </div>
              </div>

              {/* Resizable System Output Log Viewer */}
              <ResizableBlock
                id="ingest_output_log"
                defaultHeight={250}
                className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 space-y-2"
                title={
                  <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5 shrink-0">
                    <Cpu className="w-3.5 h-3.5 text-cyan-400" /> System Output Log
                  </h4>
                }
              >
                <div className="flex-1 min-h-0 bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1 overflow-y-auto">
                  {logMessages.map((msg, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <span className="text-slate-400">{`>`}</span>
                      <span>{msg}</span>
                    </div>
                  ))}
                </div>
              </ResizableBlock>
            </ResizableSplit>
          </div>
        </ResizableSplit>
      </div>
    </div>
  );
};
