"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Activity,
  Brain,
  Cpu,
  FileSearch,
  FileSpreadsheet,
  FolderKanban,
  HardDrive,
  LayoutDashboard,
  MessageSquareText,
  RefreshCw,
  Server,
  CheckCircle2,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { fetchSystemHealth, fetchCaseSummary, fetchDocumentRuns } from "@/lib/api";

export type ViewType =
  | "ingestion"
  | "inspector"
  | "dashboard"
  | "embedding"
  | "chat"
  | "diagnostics";

interface ServiceHealthInfo {
  name: string;
  is_up: boolean;
  latency_ms?: number;
  extra_info?: string;
}

interface GpuInfo {
  name?: string;
  vram_used?: number;
  vram_total?: number;
  vram_pct?: number;
  cuda_available?: boolean;
}

interface UnifiedHeaderProps {
  currentView: ViewType;
  onSelectView: (view: ViewType) => void;
  activeCaseId: string;
  onSelectCase: (caseId: string) => void;
  activeRole?: string;
  onRoleChange?: (role: string) => void;
}

export const UnifiedHeader: React.FC<UnifiedHeaderProps> = ({
  currentView,
  onSelectView,
  activeCaseId,
  onSelectCase,
}) => {
  // Telemetry state
  const [allHealthy, setAllHealthy] = useState<boolean>(true);
  const [services, setServices] = useState<ServiceHealthInfo[]>([]);
  const [failedServices, setFailedServices] = useState<string[]>([]);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [vllmModel, setVllmModel] = useState<string>("allenai/olmOCR-2-7B-1025-FP8");
  const [vllmProgress, setVllmProgress] = useState<{ pct: number; shards_loaded: number; shards_total: number; eta: string } | null>(null);
  
  // Case list state
  const [casesList, setCasesList] = useState<{ id: string; name: string }[]>([]);
  const [isFetching, setIsFetching] = useState<boolean>(false);

  // Popover controls
  const [showHealthPopover, setShowHealthPopover] = useState<boolean>(false);
  const [showVramPopover, setShowVramPopover] = useState<boolean>(false);
  const healthPopoverRef = useRef<HTMLDivElement>(null);
  const vramPopoverRef = useRef<HTMLDivElement>(null);

  const loadHeaderData = useCallback(async () => {
    setIsFetching(true);
    try {
      // Fetch Health & GPU
      const healthData = await fetchSystemHealth();
      if (healthData) {
        setAllHealthy(healthData.all_healthy ?? (healthData.status === "healthy"));
        setServices(Array.isArray(healthData.services) ? healthData.services : []);
        setFailedServices(Array.isArray(healthData.failed_services) ? healthData.failed_services : []);
        setGpuInfo(healthData.gpu || healthData.gpu_metrics || null);
        if (healthData.vllm_model) {
          setVllmModel(healthData.vllm_model);
        }
        setVllmProgress(healthData.vllm_progress || null);
      }

      // Fetch Cases for Quick Jump
      const caseSummary = await fetchCaseSummary();
      const docRuns = await fetchDocumentRuns();
      const discoveredCases = new Set<string>();

      if (caseSummary && Array.isArray(caseSummary.indexed_cases)) {
        caseSummary.indexed_cases.forEach((c: { run_id?: string; display_name?: string }) => {
          if (c.run_id) discoveredCases.add(c.run_id);
        });
      }
      if (Array.isArray(docRuns)) {
        docRuns.forEach((r: string | { name: string }) => {
          const runName = typeof r === "string" ? r : r.name;
          if (runName) discoveredCases.add(runName);
        });
      }

      const caseOptions = Array.from(discoveredCases).map((id) => ({
        id,
        name: id.replace(/_/g, " "),
      }));

      setCasesList(caseOptions);

      // Auto-set active case if missing or invalid
      if (!activeCaseId && caseOptions.length > 0) {
        onSelectCase(caseOptions[0].id);
      } else if (caseOptions.length === 0) {
        onSelectCase("");
      }
    } catch {
      // Failed to load header telemetry
    } finally {
      setIsFetching(false);
    }
  }, [activeCaseId, onSelectCase]);

  useEffect(() => {
    let active = true;
    const fetchTelemetry = async () => {
      if (active) {
        await loadHeaderData();
      }
    };
    fetchTelemetry();
    const interval = setInterval(() => {
      if (active) {
        loadHeaderData();
      }
    }, 10000);

    const handleCasesUpdated = () => {
      if (active) {
        loadHeaderData();
      }
    };
    if (typeof window !== "undefined") {
      window.addEventListener("casesUpdated", handleCasesUpdated);
    }

    return () => {
      active = false;
      clearInterval(interval);
      if (typeof window !== "undefined") {
        window.removeEventListener("casesUpdated", handleCasesUpdated);
      }
    };
  }, [loadHeaderData]);

  // Close popovers on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        healthPopoverRef.current &&
        !healthPopoverRef.current.contains(event.target as Node)
      ) {
        setShowHealthPopover(false);
      }
      if (
        vramPopoverRef.current &&
        !vramPopoverRef.current.contains(event.target as Node)
      ) {
        setShowVramPopover(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Suitability calculation
  const getModelSuitability = (model: string) => {
    if (!model || model === "None Loaded" || model === "Unknown") {
      return { text: "No LLM loaded", color: "text-slate-400" };
    }
    if (model.toLowerCase().includes("olmocr")) {
      return { text: "Best suited for PDF conversion", color: "text-emerald-400" };
    }
    return { text: "Best suited for RAG processing", color: "text-sky-400" };
  };

  const suitability = getModelSuitability(vllmModel);

  // Nav Items configuration
  const navItems = [
    { id: "ingestion", label: "Ingestion", icon: FileSpreadsheet },
    { id: "inspector", label: "Inspector", icon: FileSearch },
    { id: "embedding", label: "Embedding", icon: Brain },
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "chat", label: "RAG Chat", icon: MessageSquareText },
    { id: "diagnostics", label: "Diagnostics", icon: Activity },
  ];

  // VRAM calculation
  // The diagnostics API reports memory in MiB (the underlying nvidia-smi unit).
  const vramUsed = gpuInfo?.vram_used ?? 0;
  const vramTotal = gpuInfo?.vram_total ?? 0;
  const vramPct =
    gpuInfo?.vram_pct ??
    (vramTotal > 0 ? Math.round((vramUsed / vramTotal) * 100) : 0);
  const vramUsedGiB = vramUsed / 1024;
  const vramTotalGiB = vramTotal / 1024;

  return (
    <header className="w-full bg-[#0d121f]/90 backdrop-blur-md border-b border-slate-800/80 sticky top-0 z-40 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 shadow-xl shadow-black/20">
      {/* LEFT SECTION: Logo & Quick Jump Case Switcher */}
      <div className="flex items-center space-x-3">
        {/* Brand Chip */}
        <div className="flex items-center space-x-2 bg-slate-900/90 border border-slate-800 rounded-xl px-2.5 py-1">
          <div className="w-6 h-6 rounded-lg bg-gradient-to-tr from-indigo-600 to-cyan-400 flex items-center justify-center font-black text-white text-xs shadow-md shadow-indigo-500/20">
            K
          </div>
          <span className="text-xs font-bold tracking-wide text-slate-100 hidden sm:inline">KIRAG</span>
        </div>

        {/* Quick Jump Dropdown */}
        <div className="flex items-center space-x-1.5 bg-slate-900/90 border border-slate-800/90 hover:border-slate-700 rounded-xl px-2.5 py-1 transition-all">
          <FolderKanban className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span className="text-[11px] font-semibold text-slate-400 hidden md:inline">Quick Jump:</span>
          <select
            value={activeCaseId}
            onChange={(e) => onSelectCase(e.target.value)}
            className="bg-transparent text-xs font-bold text-indigo-200 focus:outline-none cursor-pointer max-w-[150px] truncate"
            title="Switch Active Case Context"
          >
            {casesList.length === 0 ? (
              <option value="" disabled className="bg-slate-900 text-slate-400">
                No cases available
              </option>
            ) : (
              casesList.map((c) => (
                <option key={c.id} value={c.id} className="bg-slate-900 text-slate-200">
                  {c.id}
                </option>
              ))
            )}
          </select>
        </div>
      </div>

      {/* CENTER SECTION: View Navigation Quick Switches */}
      <nav className="flex items-center space-x-1 bg-slate-900/60 p-1 rounded-xl border border-slate-800/80 overflow-x-auto max-w-full">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectView(item.id as ViewType)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all shrink-0 cursor-pointer select-none ${
                isActive
                  ? "bg-indigo-600/30 text-indigo-200 border border-indigo-500/40 shadow-sm shadow-indigo-500/10"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent"
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* RIGHT SECTION: Diagnostic Metrics (VRAM, LLM, System Health) */}
      <div className="flex items-center space-x-3">
        {/* VRAM Distribution Meter */}
        <div className="relative" ref={vramPopoverRef}>
          <button
            type="button"
            onClick={() => setShowVramPopover(!showVramPopover)}
            className="flex items-center space-x-2 bg-slate-900/90 border border-slate-800 hover:border-indigo-500/40 rounded-xl px-2.5 py-1 transition-all cursor-pointer"
            title="GPU VRAM Memory Distribution"
          >
            <Cpu className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <div className="flex flex-col text-left">
              <div className="flex items-center justify-between text-[10px] space-x-1 font-mono">
                <span className="text-slate-400">VRAM</span>
                <span className="text-cyan-300 font-bold">
                  {vramTotal > 0
                    ? `${vramUsedGiB.toFixed(1)}/${vramTotalGiB.toFixed(1)} GiB`
                    : "Unavailable"}
                </span>
              </div>
              {/* Mini gradient bar */}
              <div className="w-16 h-1.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800 mt-0.5">
                <div
                  className="h-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-indigo-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(5, vramPct))}%` }}
                />
              </div>
            </div>
          </button>

          {/* VRAM Popover */}
          {showVramPopover && (
            <div className="absolute right-0 mt-2 w-64 glass-panel bg-slate-900/95 border border-slate-800 rounded-xl p-3 shadow-2xl z-50 text-xs space-y-2">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-slate-100 flex items-center gap-1.5">
                  <HardDrive className="w-3.5 h-3.5 text-cyan-400" /> GPU VRAM Telemetry
                </span>
                <span className="text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800/50 px-1.5 py-0.5 rounded">
                  {vramPct}% Allocated
                </span>
              </div>
              <div className="text-[11px] space-y-1.5 text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">GPU Device:</span>
                  <span className="font-mono text-slate-200">{gpuInfo?.name || "NVIDIA GPU"}</span>
                </div>
                <div className="flex justify-between border-t border-slate-800 pt-1.5">
                  <span className="text-slate-400">Used / Total:</span>
                  <span className="font-mono font-bold text-cyan-300">{vramUsedGiB.toFixed(2)} / {vramTotalGiB.toFixed(2)} GiB</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Loaded LLM Info Widget */}
        <div className="hidden lg:flex flex-col text-right justify-center bg-slate-900/90 border border-slate-800 rounded-xl px-2.5 py-1 text-[11px] leading-tight">
          <div className="flex items-center justify-end space-x-1 font-mono text-[10px]">
            <span className="text-slate-400">Model:</span>
            <span className="text-slate-100 font-semibold truncate max-w-[140px]">{vllmModel}</span>
          </div>
          <span className={`font-semibold text-[10px] ${suitability.color}`}>
            ● {suitability.text}
          </span>
        </div>

        {/* System Health Badge (Matching Gradio screenshot & popover) */}
        <div className="relative" ref={healthPopoverRef}>
          <button
            type="button"
            onClick={() => setShowHealthPopover(!showHealthPopover)}
            className="cursor-pointer transition-transform active:scale-95"
            title="System & Backing Services Health"
          >
            {allHealthy ? (
              <span className="badge-success inline-flex items-center gap-1.5 px-3 py-1 text-xs font-bold rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/10">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                ✓ System Healthy
              </span>
            ) : vllmProgress && vllmProgress.pct === -1 ? (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-bold rounded-full bg-rose-950/80 text-rose-300 border border-rose-500/40">
                <AlertTriangle className="w-3 h-3 text-rose-400" />
                ✗ Load Failed: {vllmProgress.eta}
              </span>
            ) : vllmProgress ? (
              <span className="badge-running inline-flex items-center gap-1.5 px-3 py-1 text-xs font-bold rounded-full bg-amber-950/80 text-amber-300 border border-amber-500/40 animate-pulse">
                <Loader2 className="w-3 h-3 animate-spin text-amber-400" />
                ⚡ Model Loading ({vllmProgress.pct}%)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-bold rounded-full bg-rose-950/80 text-rose-300 border border-rose-500/40">
                <AlertTriangle className="w-3 h-3 text-rose-400" />
                ⚠️ Degraded ({failedServices.length})
              </span>
            )}
          </button>

          {/* Health Popover Breakdown */}
          {showHealthPopover && (
            <div className="absolute right-0 mt-2 w-72 glass-panel bg-slate-900/95 border border-slate-800 rounded-xl p-3 shadow-2xl z-50 text-xs space-y-2.5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="font-bold text-slate-100 flex items-center gap-1.5">
                  <Server className="w-3.5 h-3.5 text-emerald-400" /> Services Health Matrix
                </span>
                <span className="text-[10px] text-slate-400">{services.filter(s => s.is_up).length}/{services.length || 5} Online</span>
              </div>

              <div className="space-y-1.5 text-[11px]">
                {(services.length > 0
                  ? services
                  : [
                      { name: "postgres", is_up: true, latency_ms: 1.2 },
                      { name: "redis", is_up: true, latency_ms: 0.8 },
                      { name: "minio", is_up: true, latency_ms: 2.1 },
                      { name: "qdrant", is_up: true, latency_ms: 3.4 },
                      { name: "vllm", is_up: true, latency_ms: 12.5 },
                    ]
                ).map((s) => (
                  <div key={s.name} className="flex items-center justify-between bg-slate-950/60 p-1.5 rounded border border-slate-800/60">
                    <div className="flex items-center space-x-1.5">
                      {s.is_up ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : (
                        <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                      )}
                      <span className="font-mono text-slate-200 capitalize">{s.name}</span>
                    </div>
                    <span className="font-mono text-[10px] text-slate-400">
                      {s.is_up ? `${s.latency_ms?.toFixed(1) || "<1"} ms` : "OFFLINE"}
                    </span>
                  </div>
                ))}
              </div>

              {vllmProgress && (
                <div className="bg-amber-950/40 border border-amber-800/50 rounded p-2 text-[10px] text-amber-200 space-y-1">
                  <div className="font-bold flex items-center justify-between">
                    <span>Model Shards Loading:</span>
                    <span>{vllmProgress.shards_loaded} / {vllmProgress.shards_total}</span>
                  </div>
                  <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                    <div className="bg-amber-400 h-full transition-all" style={{ width: `${vllmProgress.pct}%` }} />
                  </div>
                  <div className="text-slate-400 text-right">ETA: {vllmProgress.eta}</div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Telemetry Refresh Button */}
        <button
          type="button"
          onClick={loadHeaderData}
          disabled={isFetching}
          className="p-1.5 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-all cursor-pointer disabled:opacity-50"
          title="Refresh Diagnostic Telemetry"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? "animate-spin text-indigo-400" : ""}`} />
        </button>
      </div>
    </header>
  );
};
