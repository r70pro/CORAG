"use client";

import React, { useState, useEffect } from "react";
import {
  Brain,
  Zap,
  RefreshCw,
  Cpu,
  Database,
  Layers,
  Save,
  Trash2,
  Upload,
  Play,
  Terminal,
} from "lucide-react";
import {
  fetchEmbeddingTelemetry,
  saveEmbeddingConfig,
  purgeVectorCache,
  fetchPipelineRuns,
  indexRun,
  indexAllRuns,
  uploadMarkdownFiles,
} from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";
import { ResizableBlock } from "@/components/ResizableBlock";

interface EmbeddingTelemetry {
  active_device: string;
  device_target: string;
  qdrant_points: number;
  collection_name: string;
  vector_dim: number;
  redis_cached_count: string | number;
}

export const EmbeddingPipeline: React.FC = () => {
  // Telemetry state
  const [telemetry, setTelemetry] = useState<EmbeddingTelemetry>({
    active_device: "CUDA GPU",
    device_target: "auto",
    qdrant_points: 0,
    collection_name: "cases",
    vector_dim: 1024,
    redis_cached_count: "N/A",
  });
  const [loadingTelemetry, setLoadingTelemetry] = useState<boolean>(false);

  // Configuration state
  const [device, setDevice] = useState<string>("auto");
  const [modelName, setModelName] = useState<string>("BAAI/bge-large-en-v1.5");
  const [batchSize, setBatchSize] = useState<number>(64);
  const [chunkSize, setChunkSize] = useState<number>(800);
  const [chunkOverlap, setChunkOverlap] = useState<number>(100);
  const [configStatus, setConfigStatus] = useState<string>("");

  // External Markdown Upload state
  const [mdFiles, setMdFiles] = useState<File[]>([]);
  const [targetCase, setTargetCase] = useState<string>("new");
  const [newCaseName, setNewCaseName] = useState<string>("");
  const [uploadStatus, setUploadStatus] = useState<string>("");

  // Batch Indexing state
  const [availableRuns, setAvailableRuns] = useState<{ display_name: string; run_dir: string }[]>([]);
  const [selectedRunDir, setSelectedRunDir] = useState<string>("");
  const [indexingStatus, setIndexingStatus] = useState<string>("");
  const [logMessages, setLogMessages] = useState<string[]>([
    "Ready for dense vector embedding & Qdrant indexing operations.",
  ]);

  const loadTelemetryData = async () => {
    setLoadingTelemetry(true);
    const data = await fetchEmbeddingTelemetry();
    if (data) {
      setTelemetry(data);
    }
    setLoadingTelemetry(false);
  };

  const loadRuns = async () => {
    const runs = await fetchPipelineRuns();
    if (runs && Array.isArray(runs)) {
      setAvailableRuns(runs);
      if (runs.length > 0) {
        setSelectedRunDir(runs[0].run_dir);
      }
    }
  };

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      const data = await fetchEmbeddingTelemetry();
      if (!isMounted) return;
      if (data) {
        setTelemetry(data);
      }
      const runs = await fetchPipelineRuns();
      if (!isMounted) return;
      if (runs && Array.isArray(runs)) {
        setAvailableRuns(runs);
        if (runs.length > 0) {
          setSelectedRunDir(runs[0].run_dir);
        }
      }
    };
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSaveConfig = async () => {
    setConfigStatus("Saving configuration...");
    const res = await saveEmbeddingConfig({
      embedding_model: modelName,
      embedding_device: device,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
      embedding_batch_size: batchSize,
    });
    setConfigStatus(res.message || (res.success ? "Saved!" : "Error saving config"));
    await loadTelemetryData();
  };

  const handlePurgeCache = async () => {
    setConfigStatus("Purging Redis vector cache...");
    const res = await purgeVectorCache();
    setConfigStatus(res.message || "Vector cache purged");
    await loadTelemetryData();
  };

  const handleUploadMarkdown = async () => {
    if (mdFiles.length === 0) {
      setUploadStatus("Please select at least one .md file.");
      return;
    }
    setUploadStatus("Uploading & indexing markdown files...");
    const res = await uploadMarkdownFiles(mdFiles, targetCase, newCaseName);
    setUploadStatus(res.message || "Upload & indexing completed.");
    setLogMessages((prev) => [...prev, `[Upload] ${res.message || "Done"}`]);
    await loadTelemetryData();
    await loadRuns();
  };

  const handleIndexRun = async () => {
    if (!selectedRunDir) {
      setIndexingStatus("Please select an OCR run to index.");
      return;
    }
    setIndexingStatus("Indexing selected run...");
    setLogMessages((prev) => [...prev, `[Indexing] Starting indexing for ${selectedRunDir}...`]);
    const res = await indexRun(selectedRunDir);
    setIndexingStatus(res.message || "Indexing completed.");
    setLogMessages((prev) => [...prev, `[Result] ${res.message || "Done"}`]);
    await loadTelemetryData();
  };

  const handleIndexAll = async () => {
    setIndexingStatus("Indexing all runs...");
    setLogMessages((prev) => [...prev, "[Indexing] Starting bulk indexing for all OCR runs..."]);
    const res = await indexAllRuns();
    setIndexingStatus(res.message || "Bulk indexing completed.");
    setLogMessages((prev) => [...prev, `[Result] ${res.message || "Done"}`]);
    await loadTelemetryData();
  };

  return (
    <div className="p-4 md:p-6 space-y-4 w-full h-full flex flex-col min-h-0 overflow-hidden">
      {/* Page Header */}
      <div className="glass-panel bg-slate-900/60 p-4 rounded-2xl border border-slate-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-lg shrink-0">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-purple-600/20 text-purple-400 border border-purple-500/30">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-wide flex items-center gap-2">
                Embedding Pipeline
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-emerald-950 text-emerald-300 border border-emerald-800/50">
                  ⚡ 50x GPU Accelerated
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Vector embedding generation, chunking strategies, and Qdrant index management
              </p>
            </div>
          </div>
        </div>

        {/* Header Stats & KPI Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Embedding LLM:</span>
            <span className="text-indigo-300 font-bold max-w-[150px] truncate">{modelName}</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Vector Store:</span>
            <span className="text-emerald-300 font-bold">Qdrant ● Connected</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Chunk Size:</span>
            <span className="text-cyan-300 font-bold">{chunkSize} tokens ({chunkOverlap} ovlp)</span>
          </div>

          <button
            type="button"
            onClick={loadTelemetryData}
            disabled={loadingTelemetry}
            className="px-3 py-1 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:text-white text-xs font-semibold flex items-center gap-1.5 cursor-pointer select-none"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingTelemetry ? "animate-spin text-indigo-400" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 w-full">
        <ResizableSplit direction="vertical" storageKey="embedding_main" initialSizes={[60, 40]} minSizes={[0, 0]}>
          {/* Top Telemetry & Config Scroll Area */}
          <div className="space-y-4 h-full min-h-0 overflow-y-auto pr-1">
            {/* Telemetry Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1">
                <div className="text-xs text-slate-400">Compute Engine</div>
                <div className="text-base font-bold text-emerald-400 flex items-center gap-1.5">
                  <Zap className="w-4 h-4" /> {telemetry.active_device || "CUDA GPU"}
                </div>
                <div className="text-[10px] text-slate-400">Device Target: {telemetry.device_target || "auto"}</div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1">
                <div className="text-xs text-slate-400">Qdrant Points Count</div>
                <div className="text-base font-bold text-sky-400 flex items-center gap-1.5">
                  <Database className="w-4 h-4" /> {telemetry.qdrant_points || 0} Points
                </div>
                <div className="text-[10px] text-slate-400">Collection: {telemetry.collection_name || "cases"}</div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1">
                <div className="text-xs text-slate-400">Vector Dimension</div>
                <div className="text-base font-bold text-teal-300 flex items-center gap-1.5">
                  <Layers className="w-4 h-4" /> {telemetry.vector_dim || 1024}-dim
                </div>
                <div className="text-[10px] text-slate-400">Metric: Cosine Similarity</div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1">
                <div className="text-xs text-slate-400">Redis Vector Cache</div>
                <div className="text-base font-bold text-pink-400 flex items-center gap-1.5">
                  <Cpu className="w-4 h-4" /> {telemetry.redis_cached_count || "N/A"}
                </div>
                <div className="text-[10px] text-slate-400">Bulk pipeline cached</div>
              </div>
            </div>

            {/* Configuration Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Hardware Acceleration Panel */}
              <div className="glass-panel p-4 rounded-2xl space-y-3 border border-slate-800">
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-indigo-400" /> Compute Engine & Hardware Acceleration
                </h3>

                <div className="space-y-3 text-xs">
                  <div>
                    <label className="block text-slate-400 mb-1">Compute Engine Device</label>
                    <select
                      value={device}
                      onChange={(e) => setDevice(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-slate-200 focus:outline-none"
                    >
                      <option value="auto">⚡ Auto (CUDA GPU when available)</option>
                      <option value="cuda">🚀 CUDA GPU Dedicated</option>
                      <option value="cpu">💻 CPU Mode</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-slate-400 mb-1">Embedding Model Name</label>
                    <select
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-slate-200 focus:outline-none"
                    >
                      <option value="BAAI/bge-large-en-v1.5">BAAI/bge-large-en-v1.5</option>
                      <option value="BAAI/bge-small-en-v1.5">BAAI/bge-small-en-v1.5</option>
                      <option value="BAAI/bge-base-en-v1.5">BAAI/bge-base-en-v1.5</option>
                      <option value="nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5</option>
                    </select>
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Embedding Batch Size</span>
                      <span className="font-mono text-indigo-300">{batchSize}</span>
                    </div>
                    <input
                      type="range"
                      min={16}
                      max={512}
                      step={16}
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      className="w-full accent-indigo-500 cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Hyperparameters Panel */}
              <div className="glass-panel p-4 rounded-2xl space-y-3 border border-slate-800 flex flex-col justify-between">
                <div className="space-y-3">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" /> Chunking & Indexing Hyperparameters
                  </h3>

                  <div className="space-y-3 text-xs">
                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Max Chunk Size (Characters)</span>
                        <span className="font-mono text-indigo-300">{chunkSize}</span>
                      </div>
                      <input
                        type="range"
                        min={200}
                        max={2000}
                        step={50}
                        value={chunkSize}
                        onChange={(e) => setChunkSize(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-slate-400 mb-1">
                        <span>Chunk Overlap (Characters)</span>
                        <span className="font-mono text-indigo-300">{chunkOverlap}</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={500}
                        step={10}
                        value={chunkOverlap}
                        onChange={(e) => setChunkOverlap(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={handleSaveConfig}
                      className="flex-1 px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-md shadow-indigo-500/20 cursor-pointer select-none"
                    >
                      <Save className="w-3.5 h-3.5 pointer-events-none" /> Save Configuration
                    </button>
                    <button
                      type="button"
                      onClick={handlePurgeCache}
                      className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs flex items-center gap-2 border border-slate-700 cursor-pointer select-none"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-rose-400 pointer-events-none" /> Clear Vector Cache
                    </button>
                  </div>
                  {configStatus && <p className="text-[11px] font-mono text-emerald-400">{configStatus}</p>}
                </div>
              </div>
            </div>

            {/* External Markdown Upload & Indexing */}
            <div className="glass-panel p-4 rounded-2xl space-y-3 border border-slate-800 relative z-10">
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <Upload className="w-4 h-4 text-indigo-400 pointer-events-none" /> Direct External Markdown Upload & Indexing
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2 text-xs">
                  <div>
                    <label className="block text-slate-400 mb-1">Select Markdown Files (.md)</label>
                    <input
                      type="file"
                      multiple
                      accept=".md"
                      onChange={(e) => setMdFiles(Array.from(e.target.files || []))}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-slate-200 cursor-pointer"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-400 mb-1">Target Case</label>
                    <select
                      value={targetCase}
                      onChange={(e) => setTargetCase(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-slate-200 cursor-pointer"
                    >
                      <option value="new">🆕 Create New Case</option>
                      {availableRuns.map((r, i) => (
                        <option key={i} value={r.run_dir}>
                          📁 {r.display_name}
                        </option>
                      ))}
                    </select>
                  </div>

                  {targetCase === "new" && (
                    <div>
                      <label className="block text-slate-400 mb-1">New Case Name</label>
                      <input
                        type="text"
                        placeholder="e.g. My Custom Case"
                        value={newCaseName}
                        onChange={(e) => setNewCaseName(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-slate-200"
                      />
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={handleUploadMarkdown}
                    className="px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-2 shadow-md shadow-indigo-500/20 cursor-pointer select-none"
                  >
                    <Upload className="w-3.5 h-3.5 pointer-events-none" /> Upload & Index Markdown
                  </button>
                </div>

                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs font-mono text-slate-300">
                  <div className="text-slate-400 font-semibold mb-1">Upload Status:</div>
                  <div>{uploadStatus || "No upload initiated."}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Resizable Batch Indexing Operations & Real-Time Console Panel */}
          <ResizableBlock
            id="embed_console"
            defaultHeight={280}
            className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 space-y-3 relative z-10"
            title={
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2 shrink-0">
                <Terminal className="w-4 h-4 text-cyan-400 pointer-events-none" /> Batch Indexing Operations & Real-Time Console
              </h3>
            }
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1 min-h-0">
              <div className="space-y-3 text-xs flex flex-col justify-center">
                <div>
                  <label className="block text-slate-400 mb-1">Select OCR Run to Index</label>
                  <select
                    value={selectedRunDir}
                    onChange={(e) => setSelectedRunDir(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-slate-200 cursor-pointer"
                  >
                    {availableRuns.map((r, i) => (
                      <option key={i} value={r.run_dir}>
                        {r.display_name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleIndexRun}
                    className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-2 shadow-md cursor-pointer select-none"
                  >
                    <Play className="w-3.5 h-3.5 pointer-events-none" /> Index Selected Run
                  </button>
                  <button
                    type="button"
                    onClick={handleIndexAll}
                    className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs flex items-center gap-2 border border-slate-700 cursor-pointer select-none"
                  >
                    <Layers className="w-3.5 h-3.5 pointer-events-none" /> Index All Runs
                  </button>
                </div>

                {indexingStatus && <p className="text-xs font-mono text-cyan-400">{indexingStatus}</p>}
              </div>

              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1 h-full min-h-0 overflow-y-auto">
                {logMessages.map((msg, i) => (
                  <div key={i} className="flex items-start gap-1.5">
                    <span className="text-slate-400">{`>`}</span>
                    <span>{msg}</span>
                  </div>
                ))}
              </div>
            </div>
          </ResizableBlock>
        </ResizableSplit>
      </div>
    </div>
  );
};
