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
  startDockerContainer,
  stopDockerContainer,
  createDockerContainer,
  shutdownDockerContainer,
  fetchSettings,
  updateSettings,
} from "@/lib/api";

export type ViewType =
  | "ingestion"
  | "inspector"
  | "dashboard"
  | "embedding"
  | "chat"
  | "diagnostics";

interface SidebarProps {
  currentView: ViewType;
  onSelectView: (view: ViewType) => void;
  activeRole: string;
  onRoleChange: (role: string) => void;
  density?: "comfortable" | "compact";
  onDensityChange?: (density: "comfortable" | "compact") => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onSelectView,
  activeRole,
  onRoleChange,
  density = "comfortable",
  onDensityChange,
}) => {
  const [dockerOpen, setDockerOpen] = useState<boolean>(true);
  const [hfToken, setHfToken] = useState<string>("");
  const [dockerModel, setDockerModel] = useState<string>("allenai/olmOCR-2-7B-1025-FP8");
  const [dockerPort, setDockerPort] = useState<number>(8000);
  const [gpuMem, setGpuMem] = useState<number>(0.8);
  const [maxLen, setMaxLen] = useState<number>(15360);
  const [dockerStatus, setDockerStatus] = useState<string>("Checking...");
  const [dockerMsg, setDockerMsg] = useState<string>("");
  const [saveStatus, setSaveStatus] = useState<string>("");

  const navItems = [
    { id: "ingestion", label: "📥 Ingestion Pipeline", icon: FileSpreadsheet },
    { id: "inspector", label: "🔍 Layout Inspector", icon: FileSearch },
    { id: "dashboard", label: "📊 Case Dashboard", icon: LayoutDashboard },
    { id: "embedding", label: "🧠 Embedding Pipeline", icon: Brain },
    { id: "chat", label: "💬 RAG Processing", icon: MessageSquareText },
    { id: "diagnostics", label: "🖥️ System Diagnostics", icon: Activity },
  ];

  const loadDockerState = async () => {
    const statusRes = await fetchDockerStatus();
    if (statusRes) {
      setDockerStatus(statusRes.status || "Unknown");
      setDockerMsg(statusRes.message || "");
    }
    const settings = await fetchSettings();
    if (settings) {
      if (settings.hf_token && settings.hf_token !== "********") setHfToken(settings.hf_token);
      if (settings.model_name) setDockerModel(settings.model_name);
      if (settings.docker_port) setDockerPort(settings.docker_port);
      if (settings.docker_gpu_mem) setGpuMem(settings.docker_gpu_mem);
      if (settings.docker_max_model_len) setMaxLen(settings.docker_max_model_len);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      const statusRes = await fetchDockerStatus();
      if (!isMounted) return;
      if (statusRes) {
        setDockerStatus(statusRes.status || "Unknown");
        setDockerMsg(statusRes.message || "");
      }
      const settings = await fetchSettings();
      if (!isMounted) return;
      if (settings) {
        if (settings.hf_token && settings.hf_token !== "********") setHfToken(settings.hf_token);
        if (settings.model_name) setDockerModel(settings.model_name);
        if (settings.docker_port) setDockerPort(settings.docker_port);
        if (settings.docker_gpu_mem) setGpuMem(settings.docker_gpu_mem);
        if (settings.docker_max_model_len) setMaxLen(settings.docker_max_model_len);
      }
    };
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleStartDocker = async () => {
    setDockerMsg("Starting container...");
    const res = await startDockerContainer({});
    setDockerMsg(res.message || "Started");
    await loadDockerState();
  };

  const handleStopDocker = async () => {
    setDockerMsg("Stopping container...");
    const res = await stopDockerContainer();
    setDockerMsg(res.message || "Stopped");
    await loadDockerState();
  };

  const handleRecreateDocker = async () => {
    setDockerMsg("Recreating container...");
    const res = await createDockerContainer({
      hf_token: hfToken,
      port: dockerPort,
      model: dockerModel,
      gpu_mem: gpuMem,
      max_model_len: maxLen,
    });
    setDockerMsg(res.message || "Recreated");
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
            const isActive = currentView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectView(item.id as ViewType)}
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
            <span>🐳 Inference Server ({dockerStatus})</span>
            {dockerOpen ? <ChevronUp className="w-3.5 h-3.5 text-slate-400 pointer-events-none" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-400 pointer-events-none" />}
          </button>

          {dockerOpen && (
            <div className="p-3 space-y-2.5 text-[11px]">
              <div className="text-[10px] text-slate-400">Manage the local GPU inference container.</div>

              <div>
                <label className="block text-slate-400 mb-0.5">Hugging Face Token</label>
                <input
                  type="password"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  placeholder="hf_..."
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-0.5">Model Name</label>
                <select
                  value={dockerModel}
                  onChange={(e) => setDockerModel(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                >
                  <option value="allenai/olmOCR-2-7B-1025-FP8">allenai/olmOCR-2-7B-1025-FP8</option>
                  <option value="nvidia/Phi-4-reasoning-plus-NVFP4">nvidia/Phi-4-reasoning-plus-NVFP4</option>
                  <option value="Qwen/Qwen2-VL-7B-Instruct">Qwen/Qwen2-VL-7B-Instruct</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-0.5">Docker Host Port</label>
                <input
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
                  <span className="font-mono text-indigo-300">{maxLen}</span>
                </div>
                <input
                  type="range"
                  min={2048}
                  max={131072}
                  step={1024}
                  value={maxLen}
                  onChange={(e) => setMaxLen(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>

              <div className="grid grid-cols-2 gap-1.5 pt-1">
                <button
                  type="button"
                  onClick={handleStartDocker}
                  className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] font-semibold flex items-center justify-center gap-1 border border-slate-700 cursor-pointer select-none"
                >
                  <Play className="w-3 h-3 text-emerald-400 pointer-events-none" /> Start
                </button>
                <button
                  type="button"
                  onClick={handleStopDocker}
                  className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] font-semibold flex items-center justify-center gap-1 border border-slate-700 cursor-pointer select-none"
                >
                  <Square className="w-3 h-3 text-amber-400 pointer-events-none" /> Stop
                </button>
              </div>

              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={handleRecreateDocker}
                  className="px-2 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-semibold flex items-center justify-center gap-1 shadow-sm cursor-pointer select-none"
                >
                  <RotateCw className="w-3 h-3 pointer-events-none" /> Recreate & Run
                </button>
                <button
                  type="button"
                  onClick={handleShutdownDocker}
                  className="px-2 py-1 rounded bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 text-[10px] font-semibold flex items-center justify-center gap-1 border border-rose-800/60 cursor-pointer select-none"
                >
                  <Power className="w-3 h-3 pointer-events-none" /> Shut Down
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
            onChange={(e) => onRoleChange(e.target.value)}
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
          IQ-RAG Workstation v1.0.0
        </div>
      </div>
    </aside>
  );
};
