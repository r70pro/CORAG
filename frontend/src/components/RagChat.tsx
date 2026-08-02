"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  MessageSquareText,
  Send,
  Bot,
  User,
  Play,
  Square,
  ChevronLeft,
  ChevronRight,
  Download,
  RefreshCw,
  Trash2,
  FileCheck,
  Terminal,
  LoaderCircle,
  Copy,
  BookOpen,
} from "lucide-react";
import {
  triggerRagChatSSE,
  startRagInfra,
  stopRagInfra,
  fetchRagInfraStatus,
  fetchCorpusStats,
  fetchPipelineRuns,
  triggerIndexRunSSE,
  triggerIndexAllRunsSSE,
  exportChatHistory,
  updateSettings,
  fetchSettings,
  deleteRagChatHistory,
} from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";

interface ChatMessage {
  id: string;
  sender: "user" | "bot";
  text: string;
  reasoning?: string;
  activity?: string;
  timestamp: string;
  verificationDetails?: {
    pageRange: string;
    physician: string;
    docTitle: string;
    refNo: string;
  }[];
}

interface ChatThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  analysisMode: string;
  activeCase: string;
}

const CHAT_THREADS_STORAGE_KEY = "kirag_rag_chat_threads_v1";

const createId = () =>
  typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `rag-${Date.now()}-${Math.random().toString(36).slice(2)}`;

const welcomeMessage = (): ChatMessage => ({
  id: createId(),
  sender: "bot",
  text: "Hello! I am your AI Medicolegal Assistant. Select an indexed case or enter a prompt below to cross-reference medical records.",
  timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
});

const createThread = (): ChatThread => {
  const now = new Date().toISOString();
  return {
    id: createId(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [welcomeMessage()],
    analysisMode: "💬 Free Q&A",
    activeCase: "",
  };
};

interface InfraStatus {
  postgres: string;
  redis: string;
  minio: string;
  qdrant: string;
}

interface CorpusStats {
  indexed_runs: number;
  indexed_documents: number;
  total_chunks: number;
  unique_authors: number;
}

interface AvailableRunItem {
  run_dir?: string;
  display_name?: string;
  run_id?: string;
  is_indexed?: boolean;
}

const ANALYSIS_MODES = [
  "🌐 General Knowledge",
  "💬 Free Q&A",
  "🧠 Expert Mode",
  "⚖️ Judge Mode",
  "📋 Timeline",
  "🏥 Injury Summary",
  "🔍 Inconsistency Finder",
  "💊 Medication Tracker",
  "🧬 Causation Analysis",
  "📈 Prognosis Analysis",
  "🧑‍💼 Work Capacity",
  "🩺 Treatment Planning",
] as const;

interface RagChatProps { activeRole?: string }

export const RagChat: React.FC<RagChatProps> = ({ activeRole = "Clinical Reviewer" }) => {
  const ragRequestRef = useRef<ReturnType<typeof triggerRagChatSSE> | null>(null);
  const reasoningLogRef = useRef<string>("");
  const [initialThread] = useState<ChatThread>(createThread);
  const sessionIdRef = useRef<string>(initialThread.id);
  const [persistenceReady, setPersistenceReady] = useState(false);
  const [threads, setThreads] = useState<ChatThread[]>([initialThread]);
  const [activeThreadId, setActiveThreadId] = useState<string>(initialThread.id);
  const [showControls, setShowControls] = useState<boolean>(true);
  const [prompt, setPrompt] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [analysisMode, setAnalysisMode] = useState<string>("💬 Free Q&A");
  const isGeneralKnowledge = analysisMode === "🌐 General Knowledge";
  const [activeCase, setActiveCase] = useState<string>("");

  // Infra state
  const [infraStatus, setInfraStatus] = useState<InfraStatus>({
    postgres: "unknown",
    redis: "unknown",
    minio: "unknown",
    qdrant: "unknown",
  });
  const [infraMsg, setInfraMsg] = useState<string>("");

  // Corpus stats & indexing
  const [corpusStats, setCorpusStats] = useState<CorpusStats>({
    indexed_runs: 0,
    indexed_documents: 0,
    total_chunks: 0,
    unique_authors: 0,
  });
  const [availableRuns, setAvailableRuns] = useState<AvailableRunItem[]>([]);
  const [selectedRunDir, setSelectedRunDir] = useState<string>("");

  const loadInfra = useCallback(async () => {
    const status = (await fetchRagInfraStatus()) as InfraStatus;
    setInfraStatus(status);
    const stats = (await fetchCorpusStats()) as CorpusStats;
    setCorpusStats(stats);
    const runs = (await fetchPipelineRuns()) as AvailableRunItem[];
    setAvailableRuns(runs || []);
    if (runs && runs.length > 0 && !selectedRunDir) {
      setSelectedRunDir(runs[0].run_dir || "");
    }
  }, [selectedRunDir]);

  const [isIndexing, setIsIndexing] = useState<boolean>(false);
  const [indexingMsg, setIndexingMsg] = useState<string>("");

  // Analysis settings
  const [modelUrl, setModelUrl] = useState<string>("http://localhost:8000/v1");
  const [modelName, setModelName] = useState<string>("nvidia/Phi-4-reasoning-plus-NVFP4");
  const [topK, setTopK] = useState<number>(8);
  const [useReranker, setUseReranker] = useState<boolean>(true);
  const [rerankerModel, setRerankerModel] = useState<string>("BAAI/bge-reranker-large");
  const [rerankerDevice, setRerankerDevice] = useState<string>("cuda");
  const [analysisSettingsLoaded, setAnalysisSettingsLoaded] = useState<boolean>(false);
  const [saveConfigStatus, setSaveConfigStatus] = useState<string>("");

  // Search filters
  const [filterDocType, setFilterDocType] = useState<string>("");
  const [filterAuthor, setFilterAuthor] = useState<string>("");
  const [filterDateFrom, setFilterDateFrom] = useState<string>("");
  const [filterDateTo, setFilterDateTo] = useState<string>("");

  // Log console
  const [logMessages, setLogMessages] = useState<string[]>([
    "RAG processing workstation ready.",
  ]);

  const [messages, setMessages] = useState<ChatMessage[]>(initialThread.messages);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [visibleSources, setVisibleSources] = useState<Set<string>>(new Set());

  useEffect(() => {
    try {
      const stored = localStorage.getItem(CHAT_THREADS_STORAGE_KEY);
      const parsed = stored ? JSON.parse(stored) : null;
      if (Array.isArray(parsed) && parsed.length > 0) {
        const validThreads = parsed.filter((thread): thread is ChatThread =>
          thread && typeof thread.id === "string" && Array.isArray(thread.messages),
        );
        if (validThreads.length > 0) {
          const mostRecent = [...validThreads].sort((a, b) =>
            String(b.updatedAt).localeCompare(String(a.updatedAt)),
          )[0];
          // Restore the external browser store only after hydration.
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setThreads(validThreads);
          setActiveThreadId(mostRecent.id);
          sessionIdRef.current = mostRecent.id;
          setMessages(mostRecent.messages);
          setAnalysisMode(mostRecent.analysisMode || "💬 Free Q&A");
          setActiveCase(mostRecent.activeCase || "");
        }
      }
    } catch {
      localStorage.removeItem(CHAT_THREADS_STORAGE_KEY);
    } finally {
      setPersistenceReady(true);
    }
  }, []);

  useEffect(() => {
    if (!persistenceReady) return;
    // Keep the thread index and its localStorage representation atomic.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setThreads((current) => {
      const now = new Date().toISOString();
      const firstUserMessage = messages.find((message) => message.sender === "user")?.text.trim();
      const updated = current.map((thread) => thread.id === activeThreadId
        ? {
            ...thread,
            title: firstUserMessage ? firstUserMessage.slice(0, 60) : "New chat",
            updatedAt: now,
            messages,
            analysisMode,
            activeCase,
          }
        : thread,
      );
      localStorage.setItem(CHAT_THREADS_STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  }, [activeCase, activeThreadId, analysisMode, messages, persistenceReady]);

  const handleIndexSelectedRun = () => {
    const targetDir = selectedRunDir || (availableRuns.length > 0 ? availableRuns[0].run_dir : "");
    if (!targetDir || isIndexing) return;
    setIsIndexing(true);
    setIndexingMsg("⏳ Indexing selected run...");
    triggerIndexRunSSE(
      targetDir,
      (message) => setIndexingMsg(message.trim()),
      (error) => {
        setIndexingMsg(`❌ Indexing error: ${String(error)}`);
        setIsIndexing(false);
      },
      () => {
        setIndexingMsg("✅ Indexing completed successfully.");
        setIsIndexing(false);
        void loadInfra();
      },
    );
  };

  const handleIndexAllRuns = () => {
    if (isIndexing) return;
    setIsIndexing(true);
    setIndexingMsg("⏳ Indexing all runs...");
    triggerIndexAllRunsSSE(
      (message) => setIndexingMsg(message.trim()),
      (error) => {
        setIndexingMsg(`❌ Bulk indexing error: ${String(error)}`);
        setIsIndexing(false);
      },
      () => {
        setIndexingMsg("✅ Bulk indexing completed successfully.");
        setIsIndexing(false);
        void loadInfra();
      },
    );
  };

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      const status = (await fetchRagInfraStatus()) as InfraStatus;
      if (!isMounted) return;
      setInfraStatus(status);
      const stats = (await fetchCorpusStats()) as CorpusStats;
      if (!isMounted) return;
      setCorpusStats(stats);
      const runs = (await fetchPipelineRuns()) as AvailableRunItem[];
      if (!isMounted) return;
      setAvailableRuns(runs || []);

      try {
        const settings = (await fetchSettings()) as Record<string, unknown>;
        if (isMounted && settings) {
          if (settings.analysis_server_url) setModelUrl(String(settings.analysis_server_url));
          else if (settings.server_url) setModelUrl(String(settings.server_url));

          if (settings.analysis_model_name) setModelName(String(settings.analysis_model_name));
          else if (settings.model_name) setModelName(String(settings.model_name));

          if (settings.retrieval_top_k) setTopK(Number(settings.retrieval_top_k));
          if (typeof settings.use_reranker === "boolean") setUseReranker(settings.use_reranker);
          if (settings.reranker_model) setRerankerModel(String(settings.reranker_model));
          if (settings.reranker_device) setRerankerDevice(String(settings.reranker_device));
        }
      } catch {
        // Fallback to default state
      } finally {
        if (isMounted) setAnalysisSettingsLoaded(true);
      }
    };
    init();

    const handleCasesUpdated = () => {
      if (isMounted) {
        loadInfra();
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
  }, [loadInfra]);

  useEffect(() => {
    return () => ragRequestRef.current?.cancel("RAG view closed");
  }, []);

  const handleStartInfra = async () => {
    setInfraMsg("Starting services...");
    const res = await startRagInfra();
    setInfraMsg(res.message || "Started");
    await loadInfra();
  };

  const handleStopInfra = async () => {
    setInfraMsg("Stopping services...");
    const res = await stopRagInfra();
    setInfraMsg(res.message || "Stopped");
    await loadInfra();
  };

  const handleSaveAnalysisConfig = async () => {
    setSaveConfigStatus("Saving...");
    const res = await updateSettings({
      analysis_server_url: modelUrl,
      analysis_model_name: modelName,
      retrieval_top_k: topK,
      use_reranker: useReranker,
      reranker_model: rerankerModel,
      reranker_device: rerankerDevice,
    });
    setSaveConfigStatus(res.message || "Configuration saved");
  };

  const handleSend = () => {
    if (!prompt.trim() || isStreaming) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: "user",
      text: prompt,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    const queryText = prompt;
    setPrompt("");
    setIsStreaming(true);
    reasoningLogRef.current = "";

    const botMsgId = (Date.now() + 1).toString();
    const botMsg: ChatMessage = {
      id: botMsgId,
      sender: "bot",
      text: "",
      activity: isGeneralKnowledge
        ? "Starting general-knowledge chat without document retrieval…"
        : "Starting RAG analysis…",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, botMsg]);

    // Token-level SSE can contain thousands of events. Batch UI updates so a
    // long answer cannot starve the final DONE callback and leave the composer
    // permanently disabled.
    let pendingText = "";
    let flushTimer: ReturnType<typeof setTimeout> | undefined;
    const flushPendingText = () => {
      if (flushTimer) clearTimeout(flushTimer);
      flushTimer = undefined;
      if (!pendingText) return;
      const text = pendingText;
      pendingText = "";
      setMessages((prev) =>
        prev.map((message) =>
          message.id === botMsgId ? { ...message, text: message.text + text } : message,
        ),
      );
    };

    ragRequestRef.current?.cancel("A new RAG request started");
    ragRequestRef.current = triggerRagChatSSE(
      {
        query: queryText,
        mode: analysisMode,
        model_url: modelUrl,
        model_name: modelName,
        top_k: topK,
        case_id: isGeneralKnowledge ? undefined : activeCase || undefined,
        doc_type: isGeneralKnowledge ? undefined : filterDocType || undefined,
        author: isGeneralKnowledge ? undefined : filterAuthor || undefined,
        date_from: isGeneralKnowledge ? undefined : filterDateFrom || undefined,
        date_to: isGeneralKnowledge ? undefined : filterDateTo || undefined,
        use_reranker: isGeneralKnowledge ? false : useReranker,
        reranker_model: isGeneralKnowledge ? undefined : rerankerModel,
        reranker_device: isGeneralKnowledge ? undefined : rerankerDevice,
        reasoning_audit: activeRole === "Admin",
        session_id: sessionIdRef.current,
        stream: true,
      },
      (chunk) => {
        pendingText += chunk;
        setMessages((prev) =>
          prev.map((message) =>
            message.id === botMsgId ? { ...message, activity: undefined } : message,
          ),
        );
        if (!flushTimer) flushTimer = setTimeout(flushPendingText, 200);
      },
      (err) => {
        flushPendingText();
        ragRequestRef.current = null;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsgId ? { ...m, text: `⚠️ Error processing query: ${String(err)}` } : m
          )
        );
        setIsStreaming(false);
      },
      () => {
        flushPendingText();
        ragRequestRef.current = null;
        setIsStreaming(false);
      },
      (status) => {
        if (status.stage !== "complete") {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === botMsgId ? { ...message, activity: status.message } : message,
            ),
          );
          setLogMessages((previous) =>
            previous.at(-1) === status.message ? previous : [...previous, status.message],
          );
        }
      },
      (reasoningChunk) => {
        reasoningLogRef.current += reasoningChunk;
        setMessages((previous) => previous.map((message) =>
          message.id === botMsgId
            ? { ...message, reasoning: (message.reasoning || "") + reasoningChunk, activity: undefined }
            : message
        ));
        setLogMessages((previous) => {
          const auditLine = `[LLM THINKING — ADMIN AUDIT]\n${reasoningLogRef.current}`;
          if (previous.at(-1)?.startsWith("[LLM THINKING — ADMIN AUDIT]")) {
            return [...previous.slice(0, -1), auditLine];
          }
          return [...previous, auditLine];
        });
      },
    );
  };

  const handleStop = useCallback(() => {
    if (!isStreaming) return;
    ragRequestRef.current?.cancel("RAG generation stopped by user");
    ragRequestRef.current = null;
    setIsStreaming(false);
    setLogMessages((previous) => [...previous, "RAG chat and model inference stopped by user."]);
  }, [isStreaming]);

  const handleClearChat = useCallback(() => {
    if (isStreaming) {
      ragRequestRef.current?.cancel("RAG chat cleared by user");
      ragRequestRef.current = null;
      setIsStreaming(false);
    }
    setMessages([]);
  }, [isStreaming]);

  const handleSelectThread = (threadId: string) => {
    const thread = threads.find((candidate) => candidate.id === threadId);
    if (!thread || thread.id === activeThreadId) return;
    ragRequestRef.current?.cancel("RAG chat thread changed");
    ragRequestRef.current = null;
    setIsStreaming(false);
    setActiveThreadId(thread.id);
    sessionIdRef.current = thread.id;
    setMessages(thread.messages);
    setAnalysisMode(thread.analysisMode || "💬 Free Q&A");
    setActiveCase(thread.activeCase || "");
    setVisibleSources(new Set());
  };

  const handleNewThread = () => {
    ragRequestRef.current?.cancel("A new RAG chat thread was created");
    ragRequestRef.current = null;
    setIsStreaming(false);
    const thread = createThread();
    setThreads((current) => [...current, thread]);
    setActiveThreadId(thread.id);
    sessionIdRef.current = thread.id;
    setMessages(thread.messages);
    setAnalysisMode(thread.analysisMode);
    setActiveCase(thread.activeCase);
    setVisibleSources(new Set());
  };

  const handleDeleteThread = () => {
    const currentThread = threads.find((thread) => thread.id === activeThreadId);
    if (!currentThread || !window.confirm(`Delete chat thread “${currentThread.title}”? This cannot be undone.`)) return;
    ragRequestRef.current?.cancel("RAG chat thread deleted");
    ragRequestRef.current = null;
    setIsStreaming(false);
    const remaining = threads.filter((thread) => thread.id !== activeThreadId);
    void deleteRagChatHistory(currentThread.id);
    const nextThread = remaining.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0] || createThread();
    const nextThreads = remaining.length > 0 ? remaining : [nextThread];
    localStorage.setItem(CHAT_THREADS_STORAGE_KEY, JSON.stringify(nextThreads));
    setThreads(nextThreads);
    setActiveThreadId(nextThread.id);
    sessionIdRef.current = nextThread.id;
    setMessages(nextThread.messages);
    setAnalysisMode(nextThread.analysisMode || "💬 Free Q&A");
    setActiveCase(nextThread.activeCase || "");
    setVisibleSources(new Set());
  };

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "n") {
        event.preventDefault();
        handleClearChat();
      } else if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "c") {
        const lastBotMessage = [...messages].reverse().find((message) => message.sender === "bot");
        if (lastBotMessage?.text) {
          event.preventDefault();
          void navigator.clipboard.writeText(lastBotMessage.text);
        }
      }
    };
    document.addEventListener("keydown", handleShortcut);
    return () => document.removeEventListener("keydown", handleShortcut);
  }, [handleClearChat, messages]);

  const handleExport = async (format: string) => {
    const historyPayload = messages.map((m) => ({
      role: m.sender === "user" ? "user" : "assistant",
      content: m.text,
      ...(m.reasoning ? { reasoning: m.reasoning } : {}),
    }));
    await exportChatHistory(historyPayload, analysisMode, activeCase, format, activeRole === "Admin");
  };

  const handleCopyResponse = async (message: ChatMessage) => {
    await navigator.clipboard.writeText(message.text);
    setCopiedMessageId(message.id);
    window.setTimeout(() => setCopiedMessageId((id) => id === message.id ? null : id), 1600);
  };

  const handleDownloadResponse = (message: ChatMessage) => {
    const blob = new Blob([message.text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `kirag-response-${message.id}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const toggleSources = (messageId: string) => {
    setVisibleSources((current) => {
      const next = new Set(current);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      return next;
    });
  };

  return (
    <div className="p-4 md:p-6 space-y-4 w-full h-full flex flex-col min-h-0 overflow-hidden">
      {/* Page Header */}
      <div className="glass-panel bg-slate-900/60 p-4 rounded-2xl border border-slate-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-lg shrink-0">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
              <MessageSquareText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-wide flex items-center gap-2">
                {isGeneralKnowledge ? "General Knowledge Chat" : "RAG Processing"}
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-indigo-950 text-indigo-300 border border-indigo-800/50">
                  {isGeneralKnowledge ? "No Document Retrieval" : "Citation Grounded"}
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                {isGeneralKnowledge
                  ? "Conversation using the analysis model's learned knowledge; indexed documents are not searched"
                  : "Multimodal RAG query engine, hybrid search, reranking, and citation-backed answer generation"}
              </p>
            </div>
          </div>
        </div>

        {/* Header Stats & KPI Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Mode:</span>
            <span className="text-indigo-300 font-bold capitalize">{analysisMode}</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Reranker:</span>
            <span className="text-emerald-300 font-bold">
              {isGeneralKnowledge
                ? "Not used"
                : useReranker ? `Active (${rerankerDevice.toUpperCase()})` : "Disabled"}
            </span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Top-K:</span>
            <span className="text-cyan-300 font-bold">
              {isGeneralKnowledge ? "Not used" : `${topK} chunks`}
            </span>
          </div>

          <button
            type="button"
            onClick={() => setShowControls(!showControls)}
            className="px-3 py-1 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:text-white text-xs font-semibold flex items-center gap-1.5 cursor-pointer select-none"
          >
            {showControls ? <ChevronLeft className="w-3.5 h-3.5 pointer-events-none" /> : <ChevronRight className="w-3.5 h-3.5 pointer-events-none" />}
            <span>{showControls ? "Hide Panel" : "Show Panel"}</span>
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 w-full">
        {showControls ? (
          <ResizableSplit direction="horizontal" storageKey="rag_chat_main" initialSizes={[30, 70]} minSizes={[0, 0]}>
            {/* Left Controls Sidebar (4 Accordions) */}
            <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3 h-full min-h-0 overflow-y-auto relative z-10">
              {/* Accordion 1: Infrastructure */}
              <div className="border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-slate-900/80 text-xs font-bold text-slate-200 flex justify-between items-center">
                  <span>🔧 RAG Infrastructure</span>
                  <span className="text-[10px] font-mono text-emerald-400">Ready</span>
                </div>
                <div className="p-3 space-y-2 text-xs">
                  <div className="text-[11px] text-slate-400">
                    Status: PG ({infraStatus.postgres || "up"}) · Redis ({infraStatus.redis || "up"}) · Qdrant ({infraStatus.qdrant || "up"})
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={handleStartInfra}
                      className="flex-1 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center justify-center gap-1 border border-slate-700 cursor-pointer select-none"
                    >
                      <Play className="w-3 h-3 text-emerald-400 pointer-events-none" /> Start
                    </button>
                    <button
                      type="button"
                      onClick={handleStopInfra}
                      className="flex-1 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center justify-center gap-1 border border-slate-700 cursor-pointer select-none"
                    >
                      <Square className="w-3 h-3 text-rose-400 pointer-events-none" /> Stop
                    </button>
                  </div>
                  {infraMsg && <p className="text-[10px] font-mono text-cyan-300">{infraMsg}</p>}
                </div>
              </div>

              {/* Accordion 2: Document Indexing */}
              <div className="border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-slate-900/80 text-xs font-bold text-slate-200 flex justify-between items-center">
                  <span>📦 Document Indexing</span>
                  <button type="button" onClick={loadInfra} className="p-1 hover:text-indigo-300 cursor-pointer">
                    <RefreshCw className="w-3 h-3 pointer-events-none" />
                  </button>
                </div>
                <div className="p-3 space-y-2 text-xs">
                  <div className="text-[11px] text-slate-300">
                    Runs: {corpusStats.indexed_runs} · Docs: {corpusStats.indexed_documents} · Chunks: {corpusStats.total_chunks}
                  </div>
                  <div>
                    <label htmlFor="rag-index-run" className="block text-slate-400 text-[10px] mb-1">Select OCR Run to Index</label>
                    <select
                      id="rag-index-run"
                      value={selectedRunDir}
                      onChange={(e) => setSelectedRunDir(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] cursor-pointer"
                    >
                      <option value="">Choose a run...</option>
                      {availableRuns.map((r, i) => (
                        <option key={i} value={r.run_dir}>
                          {r.display_name}
                        </option>
                      ))}
                    </select>
                    {selectedRunDir && (() => {
                      const sel = availableRuns.find((r) => r.run_dir === selectedRunDir);
                      const isIdx = sel?.is_indexed || sel?.display_name?.includes("[INDEXED]");
                      return (
                        <div className="mt-1 flex items-center text-[10px] font-mono">
                          {isIdx ? (
                            <span className="text-emerald-400 font-semibold bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/40">
                              ✅ Indexed in Vector Corpus
                            </span>
                          ) : (
                            <span className="text-amber-400 font-semibold bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-800/40">
                              ⚡ Pending Indexing
                            </span>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      type="button"
                      disabled={isIndexing}
                      suppressHydrationWarning
                      onClick={handleIndexSelectedRun}
                      className="flex-1 py-1 rounded bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold cursor-pointer select-none"
                    >
                      {isIndexing ? "Indexing..." : "Index Selected Run"}
                    </button>
                    <button
                      type="button"
                      disabled={isIndexing}
                      suppressHydrationWarning
                      onClick={handleIndexAllRuns}
                      className="flex-1 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 text-xs font-semibold border border-slate-700 cursor-pointer select-none"
                    >
                      {isIndexing ? "Indexing..." : "Index All Runs"}
                    </button>
                  </div>
                  {indexingMsg && (
                    <p className="text-[10px] font-mono text-cyan-300 text-center pt-1">{indexingMsg}</p>
                  )}
                </div>
              </div>

              {/* Accordion 3: Analysis Settings */}
              <div className="border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-slate-900/80 text-xs font-bold text-slate-200 flex justify-between items-center">
                  <span>⚙️ RAG Analysis & Model Settings</span>
                </div>
                <div className="p-3 space-y-2 text-xs">
                  <div>
                    <label htmlFor="analysis-model-url" className="block text-slate-400 text-[10px] mb-0.5">Model Server URL</label>
                    <input
                      id="analysis-model-url"
                      type="text"
                      value={modelUrl}
                      onChange={(e) => setModelUrl(e.target.value)}
                      disabled={!analysisSettingsLoaded}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                    />
                  </div>
                  <div>
                    <label htmlFor="analysis-model-name" className="block text-slate-400 text-[10px] mb-0.5">Model Name</label>
                    <input
                      id="analysis-model-name"
                      type="text"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      disabled={!analysisSettingsLoaded}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                    />
                  </div>
                  <div className="flex justify-between items-center text-[11px] text-slate-300 pt-1">
                    <span>Top-K Retrieval:</span>
                    <span className="font-mono text-indigo-300 font-bold">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-cyan-300">
                    Context allocation is automatic: 32K while OCR is active, full model context otherwise.
                  </p>
                  <label className="flex items-center gap-2 text-[11px] text-slate-300">
                    <input
                      type="checkbox"
                      checked={useReranker}
                      onChange={(event) => setUseReranker(event.target.checked)}
                      disabled={!analysisSettingsLoaded}
                    />
                    Enable Cross-Encoder Reranker
                  </label>
                  <div>
                    <label htmlFor="reranker-model" className="block text-slate-400 text-[10px] mb-0.5">Reranker Model</label>
                    <input
                      id="reranker-model"
                      type="text"
                      value={rerankerModel}
                      onChange={(event) => setRerankerModel(event.target.value)}
                      disabled={!analysisSettingsLoaded || !useReranker}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] disabled:opacity-50"
                    />
                  </div>
                  <div>
                    <label htmlFor="reranker-device" className="block text-slate-400 text-[10px] mb-0.5">Reranker Device</label>
                    <select
                      id="reranker-device"
                      value={rerankerDevice}
                      onChange={(event) => setRerankerDevice(event.target.value)}
                      disabled={!analysisSettingsLoaded || !useReranker}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px] disabled:opacity-50"
                    >
                      <option value="cuda">CUDA GPU</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </div>
                  <button
                    type="button"
                    onClick={handleSaveAnalysisConfig}
                    disabled={!analysisSettingsLoaded}
                    className="w-full py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 mt-1 cursor-pointer select-none"
                  >
                    Save Model Settings
                  </button>
                  {saveConfigStatus && <p className="text-[10px] font-mono text-emerald-400 text-center">{saveConfigStatus}</p>}
                </div>
              </div>

              <div className="border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-slate-900/80 text-xs font-bold text-slate-200">
                  📋 Metadata Filters
                </div>
                <div className="p-3 space-y-2 text-xs">
                  <div>
                    <label htmlFor="rag-document-type" className="block text-slate-400 text-[10px] mb-0.5">Document Type</label>
                    <select
                      id="rag-document-type"
                      value={filterDocType}
                      onChange={(event) => setFilterDocType(event.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                    >
                      <option value="">All Types</option>
                      <option value="specialist_letter">Specialist Letter</option>
                      <option value="clinical_notes">Clinical Notes</option>
                      <option value="radiology_report">Radiology Report</option>
                      <option value="physiotherapy_report">Physiotherapy Report</option>
                      <option value="medicolegal_report">Medicolegal Report</option>
                      <option value="referral_letter">Referral Letter</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="rag-author" className="block text-slate-400 text-[10px] mb-0.5">Author</label>
                    <input
                      id="rag-author"
                      type="text"
                      value={filterAuthor}
                      onChange={(event) => setFilterAuthor(event.target.value)}
                      placeholder="All authors"
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 text-[11px]"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label htmlFor="rag-date-from" className="block text-slate-400 text-[10px] mb-0.5">Date From</label>
                      <input id="rag-date-from" type="date" value={filterDateFrom} onChange={(event) => setFilterDateFrom(event.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded px-1 py-1 text-slate-200 text-[10px]" />
                    </div>
                    <div>
                      <label htmlFor="rag-date-to" className="block text-slate-400 text-[10px] mb-0.5">Date To</label>
                      <input id="rag-date-to" type="date" value={filterDateTo} onChange={(event) => setFilterDateTo(event.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded px-1 py-1 text-slate-200 text-[10px]" />
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500">Filters apply to the next query.</p>
                </div>
              </div>
            </div>

            {/* Right Main Chat Area */}
            <div className="flex flex-col h-full min-h-0 space-y-4 pl-1">
              {/* Preset Buttons & Case Selector Bar */}
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3 shrink-0 relative z-10">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-1.5 text-xs">
                    {ANALYSIS_MODES.map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setAnalysisMode(m)}
                        className={`px-3 py-1 rounded-xl text-xs font-semibold transition-all cursor-pointer select-none ${
                          analysisMode === m
                            ? "bg-indigo-600 text-white shadow-sm"
                            : "bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>

                  <select
                    value={activeCase}
                    onChange={(e) => setActiveCase(e.target.value)}
                    disabled={isGeneralKnowledge}
                    className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-1 text-xs text-slate-200 cursor-pointer"
                  >
                    <option value="">
                      {isGeneralKnowledge ? "🌐 No case context in this mode" : "📁 Select Active Case Context"}
                    </option>
                    {availableRuns.map((r, i) => (
                      <option key={i} value={r.run_id || r.run_dir || r.display_name}>
                        {r.display_name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Export Buttons */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <select aria-label="Chat thread" value={activeThreadId} onChange={(event) => handleSelectThread(event.target.value)} className="max-w-64 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-[11px] text-slate-200 cursor-pointer">
                    {[...threads].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).map((thread) => (
                      <option key={thread.id} value={thread.id}>{thread.title}</option>
                    ))}
                  </select>
                  <button type="button" onClick={handleNewThread} aria-label="New Chat" className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold text-[11px] flex items-center gap-1 border border-indigo-500/40 cursor-pointer select-none">
                    <MessageSquareText className="w-3 h-3" /> New Chat
                  </button>
                  <button type="button" onClick={handleDeleteThread} aria-label="Delete Chat Thread" className="px-2.5 py-1 rounded-lg bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 font-semibold text-[11px] flex items-center gap-1 border border-rose-800/60 cursor-pointer select-none">
                    <Trash2 className="w-3 h-3" /> Delete Thread
                  </button>
                  <button
                    type="button"
                    onClick={handleClearChat}
                    className="px-2.5 py-1 rounded-lg bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 font-semibold text-[11px] flex items-center gap-1 border border-rose-800/60 cursor-pointer select-none"
                  >
                    <Trash2 className="w-3 h-3 pointer-events-none" /> Clear Chat
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExport("md")}
                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-[11px] flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
                  >
                    <Download className="w-3 h-3 pointer-events-none" /> Export MD
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExport("txt")}
                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-[11px] flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
                  >
                    <Download className="w-3 h-3 pointer-events-none" /> TXT
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExport("csv")}
                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-[11px] flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
                  >
                    <Download className="w-3 h-3 pointer-events-none" /> CSV
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExport("docx")}
                    className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold text-[11px] flex items-center gap-1 border border-indigo-500/40 cursor-pointer select-none"
                  >
                    <Download className="w-3 h-3 text-indigo-400 pointer-events-none" /> DOCX
                  </button>
                  <button
                    type="button"
                    onClick={() => handleExport("timeline_docx")}
                    className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold text-[11px] flex items-center gap-1 border border-indigo-500/40 cursor-pointer select-none"
                  >
                    <Download className="w-3 h-3 text-indigo-400 pointer-events-none" /> Timeline DOCX
                  </button>
                </div>
              </div>

              {/* Chat Window + System Console Resizable Area */}
              <div className="flex-1 min-h-0 w-full">
                <ResizableSplit direction="vertical" storageKey="rag_chat_log_split" initialSizes={[75, 25]} minSizes={[0, 0]}>
                  {/* Main Chat Box */}
                  <div className="glass-panel rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 overflow-hidden relative z-10">
                    <div className="flex-1 min-h-0 p-5 overflow-y-auto space-y-4">
                      {messages.map((msg) => (
                        <div
                          key={msg.id}
                          className={`flex space-x-3 ${
                            msg.sender === "user" ? "justify-end" : "justify-start"
                          }`}
                        >
                          {msg.sender === "bot" && (
                            <div className="w-8 h-8 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-300 shrink-0">
                              <Bot className="w-4 h-4 pointer-events-none" />
                            </div>
                          )}

                          <div
                            className={`w-[80%] max-w-none p-4 rounded-2xl text-xs leading-relaxed ${
                              msg.sender === "user"
                                ? "bg-indigo-600 text-white shadow-md"
                                : "bg-slate-900/90 border border-slate-800 text-slate-200"
                            }`}
                          >
                            <div className="whitespace-pre-wrap">{msg.text}</div>
                            {activeRole === "Admin" && msg.reasoning && (
                              <details className="mt-3 rounded border border-amber-800/60 bg-amber-950/20 p-2">
                                <summary className="cursor-pointer text-[11px] font-semibold text-amber-300">LLM reasoning — administrative audit</summary>
                                <pre className="mt-2 whitespace-pre-wrap text-[10px] text-amber-100/80 font-mono">{msg.reasoning}</pre>
                              </details>
                            )}
                            {msg.sender === "bot" && !msg.text && msg.activity && (
                              <div className="flex items-center gap-2 text-indigo-300" role="status" aria-live="polite">
                                <LoaderCircle className="w-4 h-4 animate-spin" />
                                <span>{msg.activity}</span>
                              </div>
                            )}

                            {msg.sender === "bot" && visibleSources.has(msg.id) && (
                              <div className="mt-3 pt-3 border-t border-slate-800 space-y-1.5">
                                <div className="text-[10px] font-bold text-emerald-400 flex items-center gap-1">
                                  <FileCheck className="w-3 h-3 pointer-events-none" /> Sources
                                </div>
                                {msg.verificationDetails?.map((v, idx) => (
                                  <div
                                    key={idx}
                                    className="bg-slate-950/60 p-2 rounded-lg border border-slate-800 text-[11px] text-slate-300 flex justify-between items-center"
                                  >
                                    <div>
                                      <span className="font-semibold text-slate-200">{v.docTitle}</span> — {v.physician}
                                    </div>
                                    <span className="font-mono text-indigo-300 font-bold bg-indigo-950/40 px-2 py-0.5 rounded">
                                      {v.pageRange}
                                    </span>
                                  </div>
                                )) || <div className="text-[11px] text-slate-400">No structured source details are available for this response.</div>}
                              </div>
                            )}

                            {msg.sender === "bot" && msg.text && (
                              <div className="mt-3 pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-1.5">
                                <button type="button" onClick={() => void handleCopyResponse(msg)} aria-label="Copy Response" className="px-2 py-1 rounded-md border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center gap-1">
                                  <Copy className="w-3 h-3" /> {copiedMessageId === msg.id ? "Copied" : "Copy Response"}
                                </button>
                                <button type="button" onClick={() => handleDownloadResponse(msg)} aria-label="Download Response" className="px-2 py-1 rounded-md border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center gap-1">
                                  <Download className="w-3 h-3" /> Download Response
                                </button>
                                <button type="button" onClick={() => toggleSources(msg.id)} aria-label="Sources" aria-expanded={visibleSources.has(msg.id)} className="px-2 py-1 rounded-md border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center gap-1">
                                  <BookOpen className="w-3 h-3" /> Sources
                                </button>
                              </div>
                            )}

                            <div className="mt-2 text-[10px] text-slate-400 text-right">{msg.timestamp}</div>
                          </div>

                          {msg.sender === "user" && (
                            <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0">
                              <User className="w-4 h-4 pointer-events-none" />
                            </div>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Input Bar */}
                    <div className="p-4 border-t border-slate-800/80 bg-slate-950/60 flex items-center gap-3 shrink-0">
                      <input
                        type="text"
                        placeholder={isGeneralKnowledge
                          ? "Ask a general-knowledge question (documents are not searched)..."
                          : "Ask a medicolegal question or request an audit..."}
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && (!isStreaming || e.ctrlKey)) handleSend();
                        }}
                        aria-keyshortcuts="Control+Enter"
                        className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                      />
                      <button
                        type="button"
                        onClick={handleSend}
                        disabled={isStreaming || !prompt.trim()}
                        aria-label="Send Query"
                        suppressHydrationWarning
                        className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs flex items-center gap-2 shadow-lg shadow-indigo-500/20 cursor-pointer select-none"
                      >
                        {isStreaming ? <RefreshCw className="w-4 h-4 animate-spin pointer-events-none" /> : <Send className="w-4 h-4 pointer-events-none" />}
                        <span>{isStreaming ? "Generating…" : "Send Query"}</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleStop}
                        disabled={!isStreaming}
                        aria-label="Stop generating"
                        className="px-4 py-2.5 rounded-xl bg-rose-950/70 hover:bg-rose-900/70 disabled:opacity-40 text-rose-200 font-semibold text-xs flex items-center gap-2 border border-rose-800/70 cursor-pointer select-none"
                      >
                        <Square className="w-4 h-4 pointer-events-none" />
                        <span>Stop</span>
                      </button>
                    </div>
                  </div>

                  {/* System Log Console Accordion */}
                  <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 space-y-2">
                    <div className="text-xs font-bold text-slate-200 flex items-center gap-1.5 shrink-0">
                      <Terminal className="w-3.5 h-3.5 text-cyan-400" /> RAG System Log Console
                    </div>
                    <div className="flex-1 min-h-0 bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-y-auto space-y-1">
                      {logMessages.map((m, i) => (
                        <div key={i} className="flex items-start gap-1">
                          <span className="text-slate-400">{`>`}</span>
                          <span>{m}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </ResizableSplit>
              </div>
            </div>
          </ResizableSplit>
        ) : (
          <div className="flex flex-col h-full min-h-0 space-y-4">
            {/* Preset Buttons & Case Selector Bar */}
            <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3 shrink-0 relative z-10">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  {ANALYSIS_MODES.map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setAnalysisMode(m)}
                      className={`px-3 py-1 rounded-xl text-xs font-semibold transition-all cursor-pointer select-none ${
                        analysisMode === m
                          ? "bg-indigo-600 text-white shadow-sm"
                          : "bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>

                <select
                  value={activeCase}
                  onChange={(e) => setActiveCase(e.target.value)}
                  disabled={isGeneralKnowledge}
                  className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-1 text-xs text-slate-200 cursor-pointer"
                >
                  <option value="">
                    {isGeneralKnowledge ? "🌐 No case context in this mode" : "📁 Select Active Case Context"}
                  </option>
                  {availableRuns.map((r, i) => (
                    <option key={i} value={r.run_id || r.run_dir || r.display_name}>
                      {r.display_name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Export Buttons */}
              <div className="flex flex-wrap items-center gap-1.5">
                <select aria-label="Chat thread" value={activeThreadId} onChange={(event) => handleSelectThread(event.target.value)} className="max-w-64 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-[11px] text-slate-200 cursor-pointer">
                  {[...threads].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).map((thread) => (
                    <option key={thread.id} value={thread.id}>{thread.title}</option>
                  ))}
                </select>
                <button type="button" onClick={handleNewThread} aria-label="New Chat" className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold text-[11px] flex items-center gap-1 border border-indigo-500/40 cursor-pointer select-none">
                  <MessageSquareText className="w-3 h-3" /> New Chat
                </button>
                <button type="button" onClick={handleDeleteThread} aria-label="Delete Chat Thread" className="px-2.5 py-1 rounded-lg bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 font-semibold text-[11px] flex items-center gap-1 border border-rose-800/60 cursor-pointer select-none">
                  <Trash2 className="w-3 h-3" /> Delete Thread
                </button>
                <button
                  type="button"
                  onClick={handleClearChat}
                  className="px-2.5 py-1 rounded-lg bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 font-semibold text-[11px] flex items-center gap-1 border border-rose-800/60 cursor-pointer select-none"
                >
                  <Trash2 className="w-3 h-3 pointer-events-none" /> Clear Chat
                </button>
                <button
                  type="button"
                  onClick={() => handleExport("md")}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-[11px] flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
                >
                  <Download className="w-3 h-3 pointer-events-none" /> Export MD
                </button>
                <button
                  type="button"
                  onClick={() => handleExport("txt")}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-[11px] flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
                >
                  <Download className="w-3 h-3 pointer-events-none" /> TXT
                </button>
                <button
                  type="button"
                  onClick={() => handleExport("csv")}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-[11px] flex items-center gap-1 border border-slate-700 cursor-pointer select-none"
                >
                  <Download className="w-3 h-3 pointer-events-none" /> CSV
                </button>
                <button
                  type="button"
                  onClick={() => handleExport("docx")}
                  className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold text-[11px] flex items-center gap-1 border border-indigo-500/40 cursor-pointer select-none"
                >
                  <Download className="w-3 h-3 text-indigo-400 pointer-events-none" /> DOCX
                </button>
                <button
                  type="button"
                  onClick={() => handleExport("timeline_docx")}
                  className="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 font-semibold text-[11px] flex items-center gap-1 border border-indigo-500/40 cursor-pointer select-none"
                >
                  <Download className="w-3 h-3 text-indigo-400 pointer-events-none" /> Timeline DOCX
                </button>
              </div>
            </div>

            {/* Chat Window + System Console Resizable Area */}
            <div className="flex-1 min-h-0 w-full">
              <ResizableSplit direction="vertical" storageKey="rag_chat_log_split" initialSizes={[75, 25]} minSizes={[0, 0]}>
                {/* Main Chat Box */}
                <div className="glass-panel rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 overflow-hidden relative z-10">
                  <div className="flex-1 min-h-0 p-5 overflow-y-auto space-y-4">
                    {messages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex space-x-3 ${
                          msg.sender === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        {msg.sender === "bot" && (
                          <div className="w-8 h-8 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-300 shrink-0">
                            <Bot className="w-4 h-4 pointer-events-none" />
                          </div>
                        )}

                        <div
                          className={`w-[80%] max-w-none p-4 rounded-2xl text-xs leading-relaxed ${
                            msg.sender === "user"
                              ? "bg-indigo-600 text-white shadow-md"
                              : "bg-slate-900/90 border border-slate-800 text-slate-200"
                          }`}
                        >
                          <div className="whitespace-pre-wrap">{msg.text}</div>
                          {activeRole === "Admin" && msg.reasoning && (
                            <details className="mt-3 rounded border border-amber-800/60 bg-amber-950/20 p-2">
                              <summary className="cursor-pointer text-[11px] font-semibold text-amber-300">LLM reasoning — administrative audit</summary>
                              <pre className="mt-2 whitespace-pre-wrap text-[10px] text-amber-100/80 font-mono">{msg.reasoning}</pre>
                            </details>
                          )}
                          {msg.sender === "bot" && !msg.text && msg.activity && (
                            <div className="flex items-center gap-2 text-indigo-300" role="status" aria-live="polite">
                              <LoaderCircle className="w-4 h-4 animate-spin" />
                              <span>{msg.activity}</span>
                            </div>
                          )}

                          {msg.sender === "bot" && visibleSources.has(msg.id) && (
                            <div className="mt-3 pt-3 border-t border-slate-800 space-y-1.5">
                              <div className="text-[10px] font-bold text-emerald-400 flex items-center gap-1">
                                <FileCheck className="w-3 h-3 pointer-events-none" /> Sources
                              </div>
                              {msg.verificationDetails?.map((v, idx) => (
                                <div
                                  key={idx}
                                  className="bg-slate-950/60 p-2 rounded-lg border border-slate-800 text-[11px] text-slate-300 flex justify-between items-center"
                                >
                                  <div>
                                    <span className="font-semibold text-slate-200">{v.docTitle}</span> — {v.physician}
                                  </div>
                                  <span className="font-mono text-indigo-300 font-bold bg-indigo-950/40 px-2 py-0.5 rounded">
                                    {v.pageRange}
                                  </span>
                                </div>
                              )) || <div className="text-[11px] text-slate-400">No structured source details are available for this response.</div>}
                            </div>
                          )}

                          {msg.sender === "bot" && msg.text && (
                            <div className="mt-3 pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-1.5">
                              <button type="button" onClick={() => void handleCopyResponse(msg)} aria-label="Copy Response" className="px-2 py-1 rounded-md border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center gap-1">
                                <Copy className="w-3 h-3" /> {copiedMessageId === msg.id ? "Copied" : "Copy Response"}
                              </button>
                              <button type="button" onClick={() => handleDownloadResponse(msg)} aria-label="Download Response" className="px-2 py-1 rounded-md border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center gap-1">
                                <Download className="w-3 h-3" /> Download Response
                              </button>
                              <button type="button" onClick={() => toggleSources(msg.id)} aria-label="Sources" aria-expanded={visibleSources.has(msg.id)} className="px-2 py-1 rounded-md border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center gap-1">
                                <BookOpen className="w-3 h-3" /> Sources
                              </button>
                            </div>
                          )}

                          <div className="mt-2 text-[10px] text-slate-400 text-right">{msg.timestamp}</div>
                        </div>

                        {msg.sender === "user" && (
                          <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0">
                            <User className="w-4 h-4 pointer-events-none" />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Input Bar */}
                  <div className="p-4 border-t border-slate-800/80 bg-slate-950/60 flex items-center gap-3 shrink-0">
                    <input
                      type="text"
                      placeholder={isGeneralKnowledge
                        ? "Ask a general-knowledge question (documents are not searched)..."
                        : "Ask a medicolegal question or request an audit..."}
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && (!isStreaming || e.ctrlKey)) handleSend();
                      }}
                      aria-keyshortcuts="Control+Enter"
                      className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                    />
                    <button
                      type="button"
                      onClick={handleSend}
                      disabled={isStreaming || !prompt.trim()}
                      aria-label="Send Query"
                      className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs flex items-center gap-2 shadow-lg shadow-indigo-500/20 cursor-pointer select-none"
                    >
                      {isStreaming ? <RefreshCw className="w-4 h-4 animate-spin pointer-events-none" /> : <Send className="w-4 h-4 pointer-events-none" />}
                      <span>{isStreaming ? "Generating…" : "Send Query"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleStop}
                      disabled={!isStreaming}
                      aria-label="Stop generating"
                      className="px-4 py-2.5 rounded-xl bg-rose-950/70 hover:bg-rose-900/70 disabled:opacity-40 text-rose-200 font-semibold text-xs flex items-center gap-2 border border-rose-800/70 cursor-pointer select-none"
                    >
                      <Square className="w-4 h-4 pointer-events-none" />
                      <span>Stop</span>
                    </button>
                  </div>
                </div>

                {/* System Log Console Accordion */}
                <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col h-full min-h-0 space-y-2">
                  <div className="text-xs font-bold text-slate-200 flex items-center gap-1.5 shrink-0">
                    <Terminal className="w-3.5 h-3.5 text-cyan-400" /> RAG System Log Console
                  </div>
                  <div className="flex-1 min-h-0 bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-y-auto space-y-1">
                    {logMessages.map((m, i) => (
                      <div key={i} className="flex items-start gap-1">
                        <span className="text-slate-400">{`>`}</span>
                        <span>{m}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </ResizableSplit>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
