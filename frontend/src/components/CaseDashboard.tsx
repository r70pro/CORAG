"use client";

import React, { useState, useEffect } from "react";
import {
  FolderKanban,
  User,
  Calendar,
  Stethoscope,
  FileText,
  Layers,
  Users,
  Search,
  CheckCircle2,
  Clock,
  RefreshCw,
  Trash2,
  CheckSquare,
  Square,
  AlertTriangle,
  Filter,
} from "lucide-react";
import { fetchCaseSummary, deleteCases } from "@/lib/api";
import { ResizableSplit } from "@/components/ResizableSplit";
import { ResizableBlock } from "@/components/ResizableBlock";

interface TimelineEvent {
  date: string;
  title: string | null;
  physician: string | null;
  clinic: string | null;
  docType: string | null;
  pageRange: string | null;
  pageProvenance?: string | null;
  originalFilename?: string | null;
  refNo: string | null;
  summary: string;
}

interface CaseItem {
  run_id: string;
  client_name: string;
  dob: string;
  dob_unparsed_raw: string[];
  injuries: string[];
  documents_count: number;
  chunks_count: number;
  authors_count: number;
  date_range: string;
  indexed_at: string;
  timeline_events?: TimelineEvent[];
}

interface RawIndexedCase {
  run_id: string;
  display_name?: string;
  client_name?: string;
  dob?: string;
  dob_unparsed_raw?: string[];
  injuries?: string[];
  documents_count?: number;
  chunks_count?: number;
  authors_count?: number;
  date_range?: string;
  created_at?: string;
  indexed_at?: string;
  timeline_events?: TimelineEvent[];
}

export const CaseDashboard: React.FC = () => {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string>("");
  const [checkedIds, setCheckedIds] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [docTypeFilter, setDocTypeFilter] = useState<string>("all");
  const [loading, setLoading] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<string>("");

  const mapRawCase = (c: RawIndexedCase): CaseItem => ({
    run_id: c.run_id,
    client_name:
      c.client_name ||
      c.display_name ||
      (c.run_id ? `Case ${c.run_id.slice(0, 8)}` : "Not present in source"),
    dob: c.dob || "Not present in source",
    dob_unparsed_raw: c.dob_unparsed_raw || [],
    injuries:
      c.injuries && c.injuries.length > 0
        ? c.injuries
        : ["Not present in source"],
    documents_count: c.documents_count ?? 0,
    chunks_count: c.chunks_count ?? 0,
    authors_count: c.authors_count ?? 0,
    date_range: c.date_range || "Not present in source",
    indexed_at: c.indexed_at || c.created_at || "—",
    timeline_events: c.timeline_events || [],
  });

  const loadData = async () => {
    setLoading(true);
    const data = await fetchCaseSummary();
    if (data && Array.isArray(data.indexed_cases)) {
      if (data.indexed_cases.length > 0) {
        const fetchedCases: CaseItem[] = data.indexed_cases.map(mapRawCase);
        setCases(fetchedCases);
        if (!selectedCaseId || !fetchedCases.some((fc) => fc.run_id === selectedCaseId)) {
          setSelectedCaseId(fetchedCases[0].run_id);
        }
      } else {
        setCases([]);
        setSelectedCaseId("");
      }
    } else {
      setCases([]);
      setSelectedCaseId("");
      setStatusMsg("Unable to load case evidence; no fallback records were displayed.");
    }
    setLoading(false);
  };

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      const data = await fetchCaseSummary();
      if (!isMounted) return;
      if (data && Array.isArray(data.indexed_cases)) {
        if (data.indexed_cases.length > 0) {
          const fetchedCases: CaseItem[] = data.indexed_cases.map(mapRawCase);
          setCases(fetchedCases);
          setSelectedCaseId(fetchedCases[0].run_id);
        } else {
          setCases([]);
          setSelectedCaseId("");
        }
      } else {
        setCases([]);
        setSelectedCaseId("");
        setStatusMsg("Unable to load case evidence; no fallback records were displayed.");
      }
    };
    init();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectAll = () => {
    setCheckedIds(cases.map((c) => c.run_id));
  };

  const handleClearSelection = () => {
    setCheckedIds([]);
  };

  const toggleCheck = (runId: string, e: React.SyntheticEvent) => {
    e.stopPropagation();
    setCheckedIds((prev) =>
      prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]
    );
  };

  const handleDeleteSelected = async () => {
    if (checkedIds.length === 0) return;
    setStatusMsg(`Deleting ${checkedIds.length} selected case(s)...`);
    const res = await deleteCases(checkedIds);
    setStatusMsg(res.message || "Deleted selected cases.");
    setCheckedIds([]);
    await loadData();
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("casesUpdated"));
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm("Are you sure you want to delete ALL cases from vector store & database?")) {
      return;
    }
    setStatusMsg("Deleting all cases...");
    const res = await deleteCases([], true);
    setStatusMsg(res.message || "Deleted all cases.");
    setCheckedIds([]);
    await loadData();
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("casesUpdated"));
    }
  };

  const activeCase = cases.find((c) => c.run_id === selectedCaseId) || cases[0];
  const activeEvents = activeCase?.timeline_events || [];

  const filteredEvents = activeEvents.filter((e) => {
    const matchesSearch =
      (e.title || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (e.physician || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.summary.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDocType = docTypeFilter === "all" || e.docType === docTypeFilter;
    return matchesSearch && matchesDocType;
  });

  return (
    <div className="p-4 md:p-6 space-y-4 w-full h-full flex flex-col min-h-0 overflow-hidden">
      {/* Page Header */}
      <div className="glass-panel bg-slate-900/60 p-4 rounded-2xl border border-slate-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4 shadow-lg shrink-0">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
              <FolderKanban className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-wide flex items-center gap-2">
                Case Dashboard
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-indigo-950 text-indigo-300 border border-indigo-800/50">
                  Medicolegal Audit Active
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Medicolegal case overview, document index, patient timeline audit, and export management
              </p>
            </div>
          </div>
        </div>

        {/* Header Stats & KPI Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Case ID:</span>
            <span className="text-indigo-300 font-bold max-w-[140px] truncate">{activeCase?.run_id || "—"}</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Documents:</span>
            <span className="text-cyan-300 font-bold">{activeCase?.documents_count ?? 0} file</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Chunks:</span>
            <span className="text-emerald-300 font-bold">{activeCase?.chunks_count || 0} indexed</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-950/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs font-mono text-slate-300">
            <span className="text-slate-500">Events:</span>
            <span className="text-amber-300 font-bold">{activeEvents.length} timeline</span>
          </div>
        </div>
      </div>


      <div className="flex-1 min-h-0 w-full">
        <ResizableSplit direction="vertical" storageKey="case_dashboard_main" initialSizes={[45, 55]} minSizes={[0, 0]}>
          {/* Top Cases Grid & Action Controls */}
          <div className="space-y-4 h-full min-h-0 overflow-y-auto pr-1">
            {/* Action Controls Row */}
            <div className="flex flex-wrap items-center gap-2 text-xs relative z-10">
              <button
                type="button"
                onClick={loadData}
                disabled={loading}
                className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:text-white font-semibold flex items-center gap-1.5 cursor-pointer select-none"
              >
                <RefreshCw className={`w-3.5 h-3.5 pointer-events-none ${loading ? "animate-spin" : ""}`} />
                <span>Refresh Dashboard</span>
              </button>

              <button
                type="button"
                onClick={handleSelectAll}
                className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:text-white font-semibold flex items-center gap-1.5 cursor-pointer select-none"
              >
                <CheckSquare className="w-3.5 h-3.5 text-indigo-400 pointer-events-none" />
                <span>Select All</span>
              </button>

              <button
                type="button"
                onClick={handleClearSelection}
                className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 hover:text-white font-semibold flex items-center gap-1.5 cursor-pointer select-none"
              >
                <Square className="w-3.5 h-3.5 text-slate-400 pointer-events-none" />
                <span>Clear Selection</span>
              </button>

              <button
                type="button"
                onClick={handleDeleteSelected}
                disabled={checkedIds.length === 0}
                className="px-3 py-1.5 rounded-xl bg-rose-950/60 hover:bg-rose-900/60 disabled:opacity-40 text-rose-300 font-semibold flex items-center gap-1.5 border border-rose-800/60 shadow-sm cursor-pointer select-none"
              >
                <Trash2 className="w-3.5 h-3.5 pointer-events-none" />
                <span>Delete Selected ({checkedIds.length})</span>
              </button>

              <button
                type="button"
                onClick={handleDeleteAll}
                className="px-3 py-1.5 rounded-xl bg-rose-900 hover:bg-rose-800 text-white font-semibold flex items-center gap-1.5 shadow-sm cursor-pointer select-none"
              >
                <AlertTriangle className="w-3.5 h-3.5 pointer-events-none" />
                <span>Delete All Cases</span>
              </button>
            </div>

            {statusMsg && <div className="text-xs font-mono text-cyan-400 p-2 bg-slate-900/80 rounded-xl border border-slate-800">{statusMsg}</div>}

            {/* Case Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {cases.length === 0 ? (
                <div className="col-span-full glass-panel p-6 rounded-2xl border border-slate-800 text-center space-y-2">
                  <div className="inline-flex p-3 rounded-2xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    <FolderKanban className="w-6 h-6" />
                  </div>
                  <h3 className="text-sm font-bold text-slate-200">No Indexed Cases Found</h3>
                  <p className="text-xs text-slate-400 max-w-md mx-auto">
                    There are currently no cases indexed in the database or vector store. Upload a new document via the Ingestion Pipeline to begin medicolegal audit.
                  </p>
                </div>
              ) : (
                cases.map((c) => {
                  const isSelected = c.run_id === selectedCaseId;
                  const isChecked = checkedIds.includes(c.run_id);
                  return (
                    <div
                      key={c.run_id}
                      onClick={() => setSelectedCaseId(c.run_id)}
                      className={`glass-panel p-4 rounded-2xl cursor-pointer transition-all relative ${
                        isSelected
                          ? "border-indigo-500/80 ring-1 ring-indigo-500/50 bg-slate-900/90 shadow-lg shadow-indigo-500/10"
                          : "hover:border-slate-700 hover:bg-slate-900/60 border border-slate-800"
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={(e) => {
                              e.stopPropagation();
                              toggleCheck(c.run_id, e);
                            }}
                            className="mt-1 accent-indigo-500 w-4 h-4 rounded cursor-pointer"
                          />
                          <div>
                            <span className="text-[10px] font-mono text-slate-400">📁 {c.run_id}</span>
                            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-1.5 mt-0.5">
                              <User className="w-4 h-4 text-indigo-400" />
                              {c.client_name}
                            </h3>
                          </div>
                        </div>

                        {isSelected && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" /> Active Case
                          </span>
                        )}
                      </div>

                      <div className="mt-2 flex items-center gap-2 text-xs text-slate-300">
                        <Calendar className="w-3.5 h-3.5 text-indigo-400" />
                        <span className="font-semibold text-slate-400">DOB:</span>
                        <span>{c.dob}</span>
                      </div>
                      {c.dob_unparsed_raw.length > 0 && (
                        <div className="mt-1 text-[11px] text-amber-300">
                          Unparsed source DOB: {c.dob_unparsed_raw.join(", ")}
                        </div>
                      )}

                      <div className="mt-2 p-2 rounded-xl bg-indigo-950/20 border border-indigo-500/15">
                        <div className="text-[10px] font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1">
                          <Stethoscope className="w-3 h-3" /> Key Injury / Diagnosis
                        </div>
                        <ul className="mt-1 space-y-0.5 text-xs text-slate-200">
                          {c.injuries.map((inj, i) => (
                            <li key={i} className="flex items-center gap-1.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
                              <span>{inj}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
                        <span className="flex items-center gap-1">
                          <FileText className="w-3 h-3 text-slate-400" /> {c.documents_count} Docs
                        </span>
                        <span className="flex items-center gap-1">
                          <Layers className="w-3 h-3 text-indigo-400" /> {c.chunks_count} Chunks
                        </span>
                        <span className="flex items-center gap-1">
                          <Users className="w-3 h-3 text-cyan-400" /> {c.authors_count} Authors
                        </span>
                      </div>

                      {/* Sparkline metric visual bar */}
                      <div className="mt-2 w-full bg-slate-900 rounded-full h-1 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-indigo-500 via-cyan-400 to-emerald-400 h-full rounded-full"
                          style={{ width: `${Math.min(100, c.chunks_count * 1.5)}%` }}
                        />
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Bottom Resizable Timeline Section */}
          <ResizableBlock
            id="case_timeline"
            defaultHeight={350}
            className="glass-panel p-4 md:p-5 rounded-2xl flex flex-col h-full min-h-0 space-y-3 border border-slate-800"
            title={
              <div className="flex items-center justify-between border-b border-slate-800 pb-2 shrink-0 w-full">
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  Chronological Medicolegal Timeline: {activeCase?.client_name || "No Active Case"}
                </h3>
                <span className="text-xs font-mono text-slate-400 pr-2">
                  {activeCase?.date_range || ""}
                </span>
              </div>
            }
            headerActions={
              <div className="flex items-center space-x-2 text-xs shrink-0">
                <Filter className="w-3.5 h-3.5 text-indigo-400" />
                <span className="text-slate-400">Filter Type:</span>
                <select
                  value={docTypeFilter}
                  onChange={(e) => setDocTypeFilter(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1 text-slate-200 focus:outline-none"
                >
                  <option value="all">All Document Types</option>
                  <option value="Specialist Correspondence">Specialist Correspondence</option>
                  <option value="Imaging Report">Imaging Report</option>
                  <option value="Operation Record">Operation Record</option>
                </select>
              </div>
            }
          >
            {/* Filter & Search Bar */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 shrink-0 mb-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search timeline events, physicians, or findings..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div className="flex-1 min-h-0 overflow-y-auto pr-1">
              {activeCase ? (
                <div className="relative pl-6 space-y-4 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-gradient-to-b before:from-indigo-500 before:via-cyan-500 before:to-emerald-500">
                  {filteredEvents.map((evt, idx) => (
                    <div key={idx} className="relative group">
                      <div className="absolute -left-[23px] top-1 w-3.5 h-3.5 rounded-full bg-slate-900 border-2 border-indigo-400 group-hover:scale-125 transition-transform" />

                      <div className="glass-panel p-3.5 rounded-xl space-y-2 border border-slate-800 hover:border-indigo-500/40 transition-colors">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="px-2 py-0.5 rounded-md text-[11px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                            📅 {evt.date}
                          </span>
                          <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/30 px-2 py-0.5 rounded border border-emerald-500/20">
                            {evt.pageRange ||
                              evt.pageProvenance ||
                              "PDF page provenance not present in source"}
                          </span>
                        </div>

                        <h4 className="text-xs font-bold text-slate-100">
                          {evt.title || "Not present in source"}
                        </h4>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-1 text-[11px] text-slate-400">
                          <div><span className="font-semibold text-slate-300">Physician:</span> {evt.physician || "Not present in source"}</div>
                          <div><span className="font-semibold text-slate-300">Clinic/Facility:</span> {evt.clinic || "Not present in source"}</div>
                          <div><span className="font-semibold text-slate-300">Type:</span> {evt.docType || "Not present in source"}</div>
                          <div><span className="font-semibold text-slate-300">Reference:</span> <span className="font-mono text-indigo-300">{evt.refNo || "Not present in source"}</span></div>
                          <div><span className="font-semibold text-slate-300">File:</span> {evt.originalFilename || "Not present in source"}</div>
                        </div>

                        <p className="text-xs text-slate-300 pt-1 border-t border-slate-800/60">
                          {evt.summary}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-slate-500 italic py-4 text-center">
                  No timeline events available for empty selection.
                </div>
              )}
            </div>
          </ResizableBlock>
        </ResizableSplit>
      </div>
    </div>
  );
};
