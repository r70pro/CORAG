"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Activity,
  RefreshCw,
  Zap,
  Trash2,
  Download,
  Terminal,
  HardDrive,
  Cpu,
  Server,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
} from "lucide-react";
import {
  fetchSystemHealth,
  fetchDockerStatus,
  fetchDockerLogs,
  fetchSettings,
  stopDockerContainer,
  createDockerContainer,
  executeCleanup,
  API_BASE_URL,
} from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";

interface ServiceStatus {
  name: string;
  desc: string;
  isUp: boolean;
  latency: string;
  badge: string;
}

interface RawServiceItem {
  name: string;
  extra_info?: string;
  is_up: boolean;
  latency_ms?: number;
}

interface GPUProcessItem {
  display_name: string;
  cmdline?: string;
  pid: number;
  vram: number;
  type_text: string;
  type_badge_style?: string;
  action_text?: string;
  action_color?: string;
  is_essential?: boolean;
}

const SERVICE_DESCRIPTIONS: Record<string, string> = {
  POSTGRES: "PostgreSQL DB (Port 5432)",
  REDIS: "Redis Cache (Port 6379)",
  MINIO: "MinIO S3 Storage (Port 9000)",
  QDRANT: "Qdrant Vector DB (Port 6333)",
  VLLM: "vLLM Inference Server (Port 8000)",
};

export const SystemDiagnostics: React.FC = () => {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // GPU metrics state
  const [cudaAvailable, setCudaAvailable] = useState<boolean>(false);
  const [gpuName, setGpuName] = useState<string>("N/A");
  const [vramPct, setVramPct] = useState<number>(0);
  const [vramUsed, setVramUsed] = useState<number>(0);
  const [vramTotal, setVramTotal] = useState<number>(0);
  const [gpuProcesses, setGpuProcesses] = useState<GPUProcessItem[]>([]);

  // Docker server status & live logs
  const [dockerStatusStr, setDockerStatusStr] = useState<string>("checking");
  const [containerLogs, setContainerLogs] = useState<string>("Fetching live vLLM container logs...");
  const [activeConsoleTab, setActiveConsoleTab] = useState<"docker" | "system">("docker");
  const [autoRefreshLogs, setAutoRefreshLogs] = useState<boolean>(true);
  const [autoScrollLogs, setAutoScrollLogs] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Reset & Cleanup Manager checkboxes
  const [cleanRuns, setCleanRuns] = useState<boolean>(true);
  const [cleanTemp, setCleanTemp] = useState<boolean>(true);
  const [cleanPycache, setCleanPycache] = useState<boolean>(true);
  const [cleanHf, setCleanHf] = useState<boolean>(false);
  const [cleanupStatus, setCleanupStatus] = useState<string>("");

  // Diagnostic Log console
  const [logMessages, setLogMessages] = useState<string[]>([
    `[${new Date().toLocaleTimeString()}] System diagnostics service initialized.`,
  ]);

  const addLogMessage = (msg: string) => {
    setLogMessages((prev) => [...prev, msg]);
  };

  const loadContainerLogs = async () => {
    try {
      const res = await fetchDockerLogs(200);
      if (res && typeof res.logs === "string") {
        setContainerLogs(res.logs);
      }
    } catch (err) {
      setContainerLogs(`Error loading container logs: ${String(err)}`);
    }
  };

  const parseHealthServices = (rawServices: any): ServiceStatus[] => {
    if (!rawServices) return [];
    let items: RawServiceItem[] = [];

    if (Array.isArray(rawServices)) {
      items = rawServices;
    } else if (typeof rawServices === "object") {
      const target = rawServices.services || rawServices;
      if (Array.isArray(target)) {
        items = target;
      } else if (target && typeof target === "object") {
        items = Object.entries(target).map(([key, val]: [string, any]) => {
          if (val && typeof val === "object") {
            return {
              name: key,
              is_up: !!val.is_up,
              latency_ms: val.latency_ms ?? val.latency ?? 0,
              extra_info: val.extra_info,
            };
          }
          return {
            name: key,
            is_up: val === "healthy" || !!val,
            latency_ms: 0,
            extra_info: typeof val === "string" ? val : undefined,
          };
        });
      }
    }

    return items.map((s: RawServiceItem) => {
      const nameUpper = (s.name || "UNKNOWN").toUpperCase();
      const defaultDesc = SERVICE_DESCRIPTIONS[nameUpper] || "Backing service";
      return {
        name: nameUpper,
        desc: s.extra_info || defaultDesc,
        isUp: !!s.is_up,
        latency: `${(s.latency_ms || 0).toFixed(1)} ms`,
        badge: s.is_up ? "UP" : "DOWN",
      };
    });
  };

  const processHealthData = (health: any) => {
    if (!health) return;
    if (health.services) {
      const parsed = parseHealthServices(health.services);
      setServices(parsed);
    }
    const gpuData = health.gpu || health.gpu_metrics;
    if (gpuData) {
      setCudaAvailable(!!gpuData.cuda_available);
      setGpuName(gpuData.gpu_name || "N/A");
      setVramPct(Math.round(gpuData.vram_pct ?? 0));
      setVramUsed(Math.round(gpuData.vram_used ?? gpuData.vram_used_mb ?? 0));
      setVramTotal(Math.round(gpuData.vram_total ?? gpuData.vram_total_mb ?? 0));
      if (Array.isArray(gpuData.processes)) {
        setGpuProcesses(gpuData.processes);
      }
    }
  };

  const loadHealth = async () => {
    setLoading(true);
    const timeStr = new Date().toLocaleTimeString();
    try {
      const health = await fetchSystemHealth();
      const docStatus = await fetchDockerStatus();
      if (docStatus && docStatus.status) {
        setDockerStatusStr(docStatus.status);
      }
      processHealthData(health);
      await loadContainerLogs();
      addLogMessage(`[${timeStr}] [Refresh] Live system health & container status updated.`);
    } catch (err) {
      addLogMessage(`[${timeStr}] [Refresh Failure] ${String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      setLoading(true);
      const health = await fetchSystemHealth();
      const docStatus = await fetchDockerStatus();
      await loadContainerLogs();
      if (!isMounted) return;
      if (docStatus && docStatus.status) {
        setDockerStatusStr(docStatus.status);
      }
      processHealthData(health);
      setLoading(false);
    };
    init();

    const interval = setInterval(() => {
      if (isMounted && autoRefreshLogs) {
        loadContainerLogs();
      }
    }, 3000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [autoRefreshLogs]);

  useEffect(() => {
    if (autoScrollLogs && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [containerLogs, activeConsoleTab, autoScrollLogs]);

  const handleDockerStop = async () => {
    setLoading(true);
    const timeStr = new Date().toLocaleTimeString();
    addLogMessage(`[${timeStr}] [Docker] Triggering stop operation for vLLM container...`);
    const res = await stopDockerContainer();
    addLogMessage(`[${timeStr}] [Docker Result] ${res.message || "Stop request complete."}`);
    await loadHealth();
  };

  const handleDockerRecreate = async () => {
    setLoading(true);
    const timeStr = new Date().toLocaleTimeString();
    addLogMessage(`[${timeStr}] [Docker] Recreating vLLM inference container...`);
    const settings = await fetchSettings().catch(() => ({}));
    const res = await createDockerContainer({
      hf_token: settings?.hf_token || "",
      port: settings?.docker_port || 8000,
      model: settings?.model_name || "allenai/olmOCR-2-7B-1025-FP8",
      gpu_mem: settings?.docker_gpu_mem || 0.8,
      max_model_len: settings?.docker_max_model_len || 15360,
    });
    addLogMessage(`[${timeStr}] [Docker Result] ${res.message || "Recreate request complete."}`);
    await loadHealth();
    await loadContainerLogs();
  };

  const handleDownloadReport = () => {
    const timeStr = new Date().toLocaleTimeString();
    addLogMessage(`[${timeStr}] [Report] Downloading diagnostic report from ${API_BASE_URL}/api/diagnostics/report...`);
    const link = document.createElement("a");
    link.href = `${API_BASE_URL}/api/diagnostics/report`;
    link.download = "diagnostic_report.md";
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleCopyLogs = () => {
    const textToCopy = activeConsoleTab === "docker" ? containerLogs : logMessages.join("\n");
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExecuteCleanup = async () => {
    const components: string[] = [];
    if (cleanRuns) components.push("runs");
    if (cleanTemp) components.push("temp");
    if (cleanPycache) components.push("pycache");
    if (cleanHf) components.push("hf");

    if (components.length === 0) {
      setCleanupStatus("Select at least one component to clean.");
      return;
    }

    const timeStr = new Date().toLocaleTimeString();
    setCleanupStatus("Executing cleanup...");
    addLogMessage(`[${timeStr}] [Cleanup] Executing reset & cleanup for components: ${components.join(", ")}`);

    try {
      const res = await executeCleanup(components);
      if (res && res.success !== false) {
        const msg = res.message || "Cleanup complete.";
        setCleanupStatus(msg);
        addLogMessage(`[${timeStr}] [Cleanup Success] ${msg}`);
        await loadHealth();
      } else {
        const msg = res?.message || "Cleanup failed due to a network or server error.";
        setCleanupStatus(`Error: ${msg}`);
        addLogMessage(`[${timeStr}] [Cleanup Failed] ${msg}`);
      }
    } catch (err) {
      const msg = `Cleanup failed: ${String(err)}`;
      setCleanupStatus(msg);
      addLogMessage(`[${timeStr}] [Cleanup Error] ${msg}`);
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-4 w-full h-full flex flex-col min-h-0 overflow-hidden">
      {/* Page Header */}
      <div className="glass-panel bg-slate-900/60 p-4 rounded-2xl border border-slate-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-lg shrink-0">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-amber-600/20 text-amber-400 border border-amber-500/30">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-wide flex items-center gap-2">
                System Diagnostics
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-amber-950 text-amber-300 border border-amber-800/50">
                  Real-time Telemetry
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Infrastructure health telemetry, Docker container logs, service latency, and cleanup tools
              </p>
            </div>
          </div>
        </div>

        {/* Header Stats & KPI Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">GPU Device:</span>
            <span className="text-cyan-300 font-bold max-w-[140px] truncate">{gpuName || "NVIDIA GPU"}</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Services:</span>
            <span className="text-emerald-300 font-bold">{services.filter((s) => s.isUp).length}/{services.length || 5} Online</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">vLLM Server:</span>
            <span className="text-indigo-300 font-bold capitalize">{dockerStatusStr}</span>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <button
              type="button"
              onClick={handleDownloadReport}
              className="px-3 py-1 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold flex items-center gap-1.5 shadow-md shadow-indigo-500/20 cursor-pointer select-none"
            >
              <Download className="w-3.5 h-3.5 pointer-events-none" />
              <span>Report</span>
            </button>

            <button
              type="button"
              onClick={handleDockerStop}
              className="px-2.5 py-1 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 font-semibold border border-slate-700 cursor-pointer select-none"
            >
              Stop Container
            </button>

            <button
              type="button"
              onClick={handleDockerRecreate}
              className="px-2.5 py-1 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold border border-indigo-500/40 cursor-pointer select-none"
            >
              Recreate
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 w-full">
        <ResizableSplit direction="vertical" storageKey="diagnostics_main" initialSizes={[60, 40]} minSizes={[0, 0]}>
          {/* Top Diagnostics & GPU Grid Scroll Area */}
          <div className="space-y-4 h-full min-h-0 overflow-y-auto pr-1">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              {/* Left Column: Backing Services & Reset Manager */}
              <div className="lg:col-span-6 space-y-4">
                {/* Backing Services Health */}
                <div className="glass-panel p-4 rounded-2xl space-y-3 border border-slate-800 relative z-10">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-400 pointer-events-none" /> Backing Services Health
                  </h3>

                  {services.length === 0 ? (
                    <div className="text-xs text-slate-400 p-4 text-center bg-slate-950 rounded-xl border border-slate-800">
                      {loading ? "Fetching live services status..." : "No backing service status available."}
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {services.map((s, idx) => (
                        <div key={idx} className="bg-slate-950 p-3 rounded-xl space-y-1.5 border border-slate-800">
                          <div className="flex items-center justify-between">
                            <span
                              className={`w-2.5 h-2.5 rounded-full ${
                                s.isUp ? "bg-emerald-400 animate-pulse" : "bg-rose-500"
                              }`}
                            />
                            <span
                              className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                                s.isUp
                                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                                  : "bg-rose-500/20 text-rose-300 border-rose-500/30"
                              }`}
                            >
                              {s.badge}
                            </span>
                          </div>

                          <h4 className="text-xs font-bold text-slate-100">{s.name}</h4>
                          <p className="text-[10px] font-mono text-slate-400">{s.desc}</p>

                          <div className="pt-1.5 border-t border-slate-800/80 flex justify-between items-center text-[11px] text-slate-400">
                            <span>Latency:</span>
                            <span className="font-mono text-indigo-300 font-semibold">{s.latency}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Reset & Cleanup Manager Panel */}
                <div className="glass-panel p-4 rounded-2xl space-y-3 border border-slate-800 relative z-10">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <HardDrive className="w-4 h-4 text-amber-400 pointer-events-none" /> 🧹 Reset & Cleanup Manager
                  </h3>
                  <p className="text-xs text-slate-400">
                    Select components to clean up and reclaim disk space:
                  </p>

                  <div className="space-y-2 text-xs">
                    <label className="flex items-center space-x-2.5 bg-slate-950 p-2.5 rounded-xl border border-slate-800 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={cleanRuns}
                        onChange={(e) => setCleanRuns(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span className="text-slate-200">Output workspace runs (`workspace/run_*`)</span>
                    </label>

                    <label className="flex items-center space-x-2.5 bg-slate-950 p-2.5 rounded-xl border border-slate-800 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={cleanTemp}
                        onChange={(e) => setCleanTemp(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span className="text-slate-200">Temporary cache & uploads (`/tmp/gradio`)</span>
                    </label>

                    <label className="flex items-center space-x-2.5 bg-slate-950 p-2.5 rounded-xl border border-slate-800 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={cleanPycache}
                        onChange={(e) => setCleanPycache(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span className="text-slate-200">Python bytecode cache (`__pycache__`)</span>
                    </label>

                    <label className="flex items-center space-x-2.5 bg-slate-950 p-2.5 rounded-xl border border-slate-800 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={cleanHf}
                        onChange={(e) => setCleanHf(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span className="text-slate-200">Vector store & HuggingFace download cache</span>
                    </label>
                  </div>

                  <button
                    type="button"
                    onClick={handleExecuteCleanup}
                    className="w-full py-2 rounded-xl bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 font-semibold text-xs flex items-center justify-center gap-2 border border-rose-800/60 shadow-sm cursor-pointer select-none"
                  >
                    <Trash2 className="w-4 h-4 text-rose-400 pointer-events-none" /> Clean Selected Components
                  </button>
                  {cleanupStatus && <p className="text-xs font-mono text-cyan-400">{cleanupStatus}</p>}
                </div>
              </div>

              {/* Right Column: GPU Metrics */}
              <div className="lg:col-span-6 space-y-4">
                <div className="glass-panel p-4 rounded-2xl space-y-3 border border-slate-800">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-indigo-400" />
                      GPU VRAM Allocation & Hardware Processes
                    </h3>
                    <span
                      className={`text-xs font-mono font-bold ${
                        cudaAvailable ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {cudaAvailable ? `CUDA Available (${gpuName})` : "CPU Mode / CUDA Unavailable"}
                    </span>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-slate-300">VRAM Usage: {vramPct}%</span>
                      <span className="font-mono text-indigo-300">
                        {vramUsed.toLocaleString()} MB / {vramTotal > 0 ? vramTotal.toLocaleString() : "N/A"} MB
                      </span>
                    </div>

                    <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800">
                      <div
                        className="bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, Math.max(0, vramPct))}%` }}
                      />
                    </div>
                  </div>

                  {/* GPU Process Table */}
                  <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 font-semibold">
                        <tr>
                          <th className="p-2.5">Process / PID</th>
                          <th className="p-2.5">VRAM Allocated</th>
                          <th className="p-2.5">Type</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-slate-200">
                        {gpuProcesses.length > 0 ? (
                          gpuProcesses.map((proc, idx) => (
                            <tr key={idx}>
                              <td className="p-2.5 font-semibold">
                                {proc.display_name} <span className="text-slate-400 font-mono text-[10px]">(PID {proc.pid})</span>
                              </td>
                              <td className="p-2.5 font-mono text-indigo-300">{proc.vram.toLocaleString()} MB</td>
                              <td className="p-2.5">
                                <span
                                  className="px-2 py-0.5 rounded text-[10px] font-bold"
                                  style={{
                                    background: proc.type_badge_style
                                      ? proc.type_badge_style.match(/background:\s*([^;]+)/)?.[1]
                                      : "rgba(99, 102, 241, 0.2)",
                                    color: proc.type_badge_style
                                      ? proc.type_badge_style.match(/color:\s*([^;]+)/)?.[1]
                                      : "#a5b4fc",
                                  }}
                                >
                                  {proc.type_text}
                                </span>
                              </td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={3} className="p-3 text-center text-slate-400 text-xs italic">
                              No active GPU hardware processes detected.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Resizable Diagnostic Console with Live Container Logs */}
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 space-y-2 relative">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2 shrink-0">
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={() => setActiveConsoleTab("docker")}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                    activeConsoleTab === "docker"
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20"
                      : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                  }`}
                >
                  🐳 Live Container Logs ({dockerStatusStr.toUpperCase()})
                </button>
                <button
                  type="button"
                  onClick={() => setActiveConsoleTab("system")}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                    activeConsoleTab === "system"
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20"
                      : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                  }`}
                >
                  🖥️ Diagnostic System Log
                </button>
              </div>

              {/* Console Toolbar Controls */}
              <div className="flex items-center space-x-2 text-[11px]">
                {activeConsoleTab === "docker" && (
                  <>
                    <label className="flex items-center space-x-1.5 text-slate-400 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={autoRefreshLogs}
                        onChange={(e) => setAutoRefreshLogs(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span>Auto-Refresh</span>
                    </label>
                    <label className="flex items-center space-x-1.5 text-slate-400 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={autoScrollLogs}
                        onChange={(e) => setAutoScrollLogs(e.target.checked)}
                        className="accent-indigo-500 rounded cursor-pointer"
                      />
                      <span>Auto-Scroll</span>
                    </label>
                  </>
                )}

                <button
                  type="button"
                  onClick={handleCopyLogs}
                  className="px-2 py-1 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 flex items-center gap-1 font-semibold cursor-pointer"
                  title="Copy log text"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-slate-400" />}
                  <span>{copied ? "Copied" : "Copy"}</span>
                </button>

                {activeConsoleTab === "docker" && (
                  <button
                    type="button"
                    onClick={loadContainerLogs}
                    className="p-1 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded border border-slate-700 cursor-pointer"
                    title="Refresh Container Logs"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-indigo-400" />
                  </button>
                )}
              </div>
            </div>

            {/* Resizable Log View Window */}
            <div
              ref={logContainerRef}
              className="flex-1 min-h-0 bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-y-auto whitespace-pre-wrap break-all leading-relaxed"
            >
              {activeConsoleTab === "docker" ? (
                containerLogs ? (
                  containerLogs
                ) : (
                  <span className="text-slate-500 italic">No logs available for container 'olmocr'.</span>
                )
              ) : (
                logMessages.map((msg, i) => (
                  <div key={i} className="flex items-start gap-1.5">
                    <span className="text-slate-500 select-none">{`>`}</span>
                    <span>{msg}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </ResizableSplit>
      </div>
    </div>
  );
};
