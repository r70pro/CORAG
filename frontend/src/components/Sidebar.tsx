"use client";

import React, { useState, useEffect } from "react";
import {
  LayoutDashboard,
  FileSearch,
  MessageSquareText,
  FileSpreadsheet,
  Activity,
  Brain,
  UserCheck,
  ChevronDown,
  ChevronUp,
  Play,
  Square,
  RotateCw,
  Power,
  Save,
} from "lucide-react";
import {
  fetchDockerStatus,
  fetchDockerModels,
  createDockerContainer,
  shutdownDockerContainer,
  fetchSettings,
  updateSettings,
  setVllmRoleRunning,
} from "@/lib/api";

export type ViewType =
  | "ingestion"
  | "inspector"
  | "dashboard"
  | "embedding"
  | "chat"
  | "diagnostics";

interface SidebarProps {
  currentView?: ViewType;
  activeView?: ViewType;
  onSelectView?: (view: ViewType) => void;
  onViewChange?: (view: ViewType) => void;
  activeRole?: string;
  onRoleChange?: (role: string) => void;
  density?: "comfortable" | "compact";
  onDensityChange?: (density: "comfortable" | "compact") => void;
}

const DEFAULT_MODEL_MAX_LENGTHS: Record<string, number> = {
  "allenai/olmOCR-2-7B-1025-FP8": 131072,
  "Qwen/Qwen3.6-35B-A3B": 262144,
  "nvidia/Phi-4-reasoning-plus-NVFP4": 32768,
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": 1048576,
  "nvidia/Llama-3.3-70B-Instruct-NVFP4": 131072,
  "openai/gpt-oss-120b": 131072,
  "google/gemma-4-31B-it": 262144,
  "Qwen/Qwen2-VL-7B-Instruct": 32768,
};

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  activeView,
  onSelectView,
  onViewChange,
  activeRole,
  onRoleChange,
  density = "comfortable",
  onDensityChange,
}) => {
  const selectedView = currentView || activeView || "ingestion";
  const handleViewChange = (v: ViewType) => {
    if (onSelectView) onSelectView(v);
    if (onViewChange) onViewChange(v);
  };

  const [dockerOpen, setDockerOpen] = useState<boolean>(true);
  const [hfToken, setHfToken] = useState<string>("");
  const [dockerModel, setDockerModel] = useState<string>("allenai/olmOCR-2-7B-1025-FP8");
  const [availableModels, setAvailableModels] = useState<string[]>([
    "allenai/olmOCR-2-7B-1025-FP8",
    "nvidia/Phi-4-reasoning-plus-NVFP4",
    "Qwen/Qwen2-VL-7B-Instruct",
  ]);
  const [modelMaxLengths, setModelMaxLengths] = useState<Record<string, number>>(DEFAULT_MODEL_MAX_LENGTHS);
  const [dockerPort, setDockerPort] = useState<number>(8000);
  const [gpuMem, setGpuMem] = useState<number>(0.8);
  const [maxLen, setMaxLen] = useState<number>(15360);
  const [tensorParallel, setTensorParallel] = useState<number>(1);
  const [dockerStatuses, setDockerStatuses] = useState<Record<"ocr" | "analysis", string>>({ ocr: "checking", analysis: "checking" });
  const [dockerMsg, setDockerMsg] = useState<string>("");
  const [saveStatus, setSaveStatus] = useState<string>("");

  const currentMaxBoundary = modelMaxLengths[dockerModel] || 131072;

  const handleModelChange = (newModel: string) => {
    setDockerModel(newModel);
    const boundary = modelMaxLengths[newModel] || 131072;
    if (maxLen > boundary) {
      setMaxLen(boundary);
    }
  };

  const navItems = [
    { id: "ingestion", label: "📥 Ingestion Pipeline", icon: FileSpreadsheet },
    { id: "inspector", label: "🔍 Layout Inspector", icon: FileSearch },
    { id: "embedding", label: "🧠 Embedding Pipeline", icon: Brain },
    { id: "dashboard", label: "📊 Case Dashboard", icon: LayoutDashboard },
    { id: "chat", label: "💬 RAG Processing", icon: MessageSquareText },
    { id: "diagnostics", label: "🖥️ System Diagnostics", icon: Activity },
  ];

  const loadDockerState = async () => {
    const [ocrStatus, analysisStatus] = await Promise.all([fetchDockerStatus("ocr"), fetchDockerStatus("analysis")]);
    setDockerStatuses({ ocr: ocrStatus?.status || "unknown", analysis: analysisStatus?.status || "unknown" });
    setDockerMsg(ocrStatus?.message || analysisStatus?.message || "");
    const modelsRes = await fetchDockerModels();
    if (modelsRes) {
      if (modelsRes.models && modelsRes.models.length > 0) {
        setAvailableModels((prev) => Array.from(new Set([...modelsRes.models, ...prev])));
      }
      if (modelsRes.max_lengths) {
        setModelMaxLengths((prev) => ({ ...prev, ...modelsRes.max_lengths }));
      }
    }
    const settings = await fetchSettings();
    if (settings) {
      if (settings.hf_token && settings.hf_token !== "********") setHfToken(settings.hf_token);
      if (settings.model_name) setDockerModel(settings.model_name);
      if (settings.docker_port) setDockerPort(settings.docker_port);
      if (settings.docker_gpu_mem) setGpuMem(settings.docker_gpu_mem);
      if (settings.docker_max_model_len) setMaxLen(settings.docker_max_model_len);
      if (settings.docker_tensor_parallel) setTensorParallel(settings.docker_tensor_parallel);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      const [statusRes, analysisStatus] = await Promise.all([fetchDockerStatus("ocr"), fetchDockerStatus("analysis")]);
      if (!isMounted) return;
      if (statusRes) {
        setDockerStatuses({ ocr: statusRes.status || "unknown", analysis: analysisStatus?.status || "unknown" });
        setDockerMsg(statusRes.message || "");
      }
      const modelsRes = await fetchDockerModels();
      if (modelsRes && isMounted) {
        if (modelsRes.models && modelsRes.models.length > 0) {
          setAvailableModels((prev) => Array.from(new Set([...modelsRes.models, ...prev])));
        }
        if (modelsRes.max_lengths) {
          setModelMaxLengths((prev) => ({ ...prev, ...modelsRes.max_lengths }));
        }
      }
      const settings = await fetchSettings();
      if (!isMounted) return;
      if (settings) {
        if (settings.hf_token && settings.hf_token !== "********") setHfToken(settings.hf_token);
        if (settings.model_name) setDockerModel(settings.model_name);
        if (settings.docker_port) setDockerPort(settings.docker_port);
        if (settings.docker_gpu_mem) setGpuMem(settings.docker_gpu_mem);
        if (settings.docker_max_model_len) setMaxLen(settings.docker_max_model_len);
        if (settings.docker_tensor_parallel) setTensorParallel(settings.docker_tensor_parallel);
      }
    };
    init();
    return () => {
      isMounted = false;
    };
  }, []);


  const pollStatusUntilDone = (maxAttempts = 15) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      const statusRes = await fetchDockerStatus("ocr");
      if (statusRes) {
        setDockerStatuses((prev) => ({ ...prev, ocr: statusRes.status || "unknown" }));
        setDockerMsg(statusRes.message || "");
        if (
          statusRes.status === "ready" ||
          statusRes.status === "stopped" ||
          statusRes.status === "error" ||
          attempts >= maxAttempts
        ) {
          clearInterval(interval);
        }
      }
    }, 2000);
  };

  const handleStartDocker = async () => {
    setDockerMsg("Starting container...");
    setDockerStatuses((prev) => ({ ...prev, ocr: "starting" }));
    const res = await setVllmRoleRunning("ocr", true);
    setDockerMsg(res.message || "Started");
    pollStatusUntilDone(15);
  };

  const handleStopDocker = async () => {
    setDockerMsg("Stopping container...");
    const res = await setVllmRoleRunning("ocr", false);
    setDockerMsg(res.message || "Stopped");
    await loadDockerState();
  };

  const handleRecreateDocker = async () => {
    setDockerMsg("Recreating container & loading model...");
    setDockerStatuses((prev) => ({ ...prev, ocr: "starting" }));
    const res = await createDockerContainer({
      hf_token: hfToken && hfToken !== "********" ? hfToken : undefined,
      port: dockerPort,
      model: dockerModel,
      gpu_mem: gpuMem,
      max_model_len: maxLen,
      tensor_parallel_size: tensorParallel,
    });
    setDockerMsg(res.message || "Recreated");
    if (res.success) {
      pollStatusUntilDone(20);
    } else {
      await loadDockerState();
    }
  };

  const handleAnalysisLifecycle = async (running: boolean) => {
    setDockerStatuses((prev) => ({ ...prev, analysis: running ? "starting" : "stopping" }));
    const res = await setVllmRoleRunning("analysis", running);
    setDockerMsg(res.message || `Analysis vLLM ${running ? "start" : "stop"} requested.`);
    await loadDockerState();
  };


  const handleShutdownDocker = async () => {
    setDockerMsg("Shutting down container...");
    const res = await shutdownDockerContainer();
    setDockerMsg(res.message || "Shutdown");
    await loadDockerState();
  };

  const handleSaveSettings = async () => {
    setSaveStatus("Saving...");
    const res = await updateSettings({
      hf_token: hfToken,
      model_name: dockerModel,
      docker_port: dockerPort,
      docker_gpu_mem: gpuMem,
      docker_max_model_len: maxLen,
      docker_tensor_parallel: tensorParallel,
    });
    setSaveStatus(res.message || "Saved");
  };

  return (
    <aside className="w-72 glass-panel h-screen flex flex-col justify-between p-4 border-r border-slate-800 shrink-0 overflow-y-auto">
      <div className="space-y-4">
        {/* Brand Header */}
        <div className="flex items-center space-x-3 px-2 py-1 border-b border-slate-800 pb-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">
            K
          </div>
          <div>
            <h1 className="font-bold text-slate-100 text-sm tracking-wide">IQ-RAG Client</h1>
            <p className="text-[11px] text-indigo-400 font-medium">Mission Control</p>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="space-y-1">
          {navItems.map((item) => {
            const isActive = selectedView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleViewChange(item.id as ViewType)}
                className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer select-none ${
                  isActive
                    ? "bg-indigo-600/25 text-indigo-200 border border-indigo-500/40 shadow-md shadow-indigo-500/10"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <span>{item.label}</span>
              </button>
            );
          })}

        </nav>

        {/* Inference Server (Docker) Accordion */}
        <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden relative z-10">
          <button
            type="button"
            onClick={() => setDockerOpen(!dockerOpen)}
            className="w-full px-3 py-2 bg-slate-900/80 flex items-center justify-between text-xs font-bold text-slate-200 cursor-pointer select-none"
          >
            <span>🐳 Dedicated vLLM Roles</span>
            {dockerOpen ? <ChevronUp className="w-3.5 h-3.5 text-slate-400 pointer-events-none" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400 pointer-events-none" />}
          </button>

          {dockerOpen && (
            <div className="p-3 space-y-2.5 text-[11px]">
              <div className="space-y-1.5">
                {(["ocr", "analysis"] as const).map((role) => (
                  <div key={role} className="rounded-lg border border-slate-800 bg-slate-950/70 p-2 space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-200 uppercase">{role} vLLM <span className="text-slate-500 font-mono">:{role === "ocr" ? "8000" : "8002"}</span></span>
                      <span className={`font-mono ${dockerStatuses[role] === "ready" ? "text-emerald-400" : dockerStatuses[role] === "starting" ? "text-amber-400" : "text-rose-300"}`}>{dockerStatuses[role]}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      <button type="button" onClick={() => role === "ocr" ? handleStartDocker() : handleAnalysisLifecycle(true)} className="px-2 py-1 rounded bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 border border-emerald-800/50 flex items-center justify-center gap-1"><Play className="w-3 h-3" /> Start</button>
                      <button type="button" onClick={() => role === "ocr" ? handleStopDocker() : handleAnalysisLifecycle(false)} className="px-2 py-1 rounded bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 border border-rose-800/50 flex items-center justify-center gap-1"><Square className="w-3 h-3" /> Stop</button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-[10px] text-slate-400 border-t border-slate-800 pt-2">OCR provisioning settings (analysis is managed independently by the production stack).</div>

              <div>
                <label htmlFor="docker-hf-token" className="block text-slate-400 mb-0.5">Hugging Face Token</label>
                <input
                  id="docker-hf-token"
                  type="password"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  placeholder="hf_..."
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-0.5">
                  <span>Tensor Parallel GPUs</span>
                  <span className="font-mono text-indigo-300">{tensorParallel}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={8}
                  step={1}
                  value={tensorParallel}
                  onChange={(e) => setTensorParallel(Number(e.target.value))}
                  aria-label="Tensor Parallel GPUs"
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div>
                <label htmlFor="docker-model" className="block text-slate-400 mb-0.5">Model Name</label>
                <select
                  id="docker-model"
                  value={dockerModel}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="docker-port" className="block text-slate-400 mb-0.5">Docker Host Port</label>
                <input
                  id="docker-port"
                  type="number"
                  value={dockerPort}
                  onChange={(e) => setDockerPort(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-0.5">
                  <span>GPU Memory Utilization</span>
                  <span className="font-mono text-indigo-300">{gpuMem}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={gpuMem}
                  onChange={(e) => setGpuMem(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-0.5">
                  <span>Max Content Length</span>
                  <span className="font-mono text-indigo-300">
                    {Math.min(maxLen, currentMaxBoundary)}{" "}
                    <span className="text-[9px] text-slate-500">(max {currentMaxBoundary.toLocaleString()})</span>
                  </span>
                </div>
                <input
                  type="range"
                  min={2048}
                  max={currentMaxBoundary}
                  step={1024}
                  value={Math.min(maxLen, currentMaxBoundary)}
                  onChange={(e) => setMaxLen(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>


              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={handleRecreateDocker}
                  className="px-2 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-semibold flex items-center justify-center gap-1 shadow-sm cursor-pointer select-none"
                >
                  <RotateCw className="w-3 h-3 pointer-events-none" /> Recreate OCR
                </button>
                <button
                  type="button"
                  onClick={handleShutdownDocker}
                  className="px-2 py-1 rounded bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 text-[10px] font-semibold flex items-center justify-center gap-1 border border-rose-800/60 cursor-pointer select-none"
                >
                  <Power className="w-3 h-3 pointer-events-none" /> Remove OCR
                </button>
              </div>

              {dockerMsg && <div className="text-[10px] font-mono text-cyan-300 pt-1">{dockerMsg}</div>}
            </div>
          )}
        </div>
      </div>

      {/* Footer Controls & Role Selection */}
      <div className="space-y-3 pt-3 border-t border-slate-800/80 relative z-10">
        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-slate-400 flex items-center space-x-1">
            <UserCheck className="w-3 h-3 text-indigo-400 pointer-events-none" />
            <span>Active Role:</span>
          </label>
          <select
            value={activeRole}
            onChange={(e) => onRoleChange && onRoleChange(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-200 focus:outline-none cursor-pointer"
          >

            <option value="Admin">Admin</option>
            <option value="Clinical Reviewer">Clinical Reviewer</option>
            <option value="Legal Specialist">Legal Specialist</option>
          </select>
        </div>

        {/* Density Toggles */}
        <div className="flex items-center space-x-1.5 text-[11px]">
          <button
            type="button"
            onClick={() => onDensityChange && onDensityChange("comfortable")}
            className={`flex-1 py-1 rounded-lg font-semibold border transition-all cursor-pointer select-none ${
              density === "comfortable"
                ? "bg-indigo-600/30 text-indigo-200 border-indigo-500/50"
                : "bg-slate-900/80 text-slate-400 border-slate-800 hover:text-slate-200"
            }`}
          >
            Comfortable
          </button>
          <button
            type="button"
            onClick={() => onDensityChange && onDensityChange("compact")}
            className={`flex-1 py-1 rounded-lg font-semibold border transition-all cursor-pointer select-none ${
              density === "compact"
                ? "bg-indigo-600/30 text-indigo-200 border-indigo-500/50"
                : "bg-slate-900/80 text-slate-400 border-slate-800 hover:text-slate-200"
            }`}
          >
            Compact
          </button>
        </div>

        <button
          type="button"
          onClick={handleSaveSettings}
          className="w-full py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold flex items-center justify-center gap-1.5 border border-slate-700 cursor-pointer select-none"
        >
          <Save className="w-3.5 h-3.5 pointer-events-none" /> Save Settings
        </button>

        {saveStatus && <p className="text-[10px] font-mono text-emerald-400 text-center">{saveStatus}</p>}

        <div className="text-[10px] text-slate-400 text-center font-mono pt-1">
          IQ-RAG Workstation v2.0.3
        </div>
      </div>
    </aside>
  );
};
