"use client";

import React, { useState } from "react";
import { Sidebar, ViewType } from "@/components/Sidebar";
import { UnifiedHeader } from "@/components/UnifiedHeader";
import { CaseDashboard } from "@/components/CaseDashboard";
import { PdfInspector } from "@/components/PdfInspector";
import { RagChat } from "@/components/RagChat";
import { IngestionPipeline } from "@/components/IngestionPipeline";
import { EmbeddingPipeline } from "@/components/EmbeddingPipeline";
import { SystemDiagnostics } from "@/components/SystemDiagnostics";
import { ErrorBoundary } from "@/components/ErrorBoundary";

export default function Home() {
  const [currentView, setCurrentView] = useState<ViewType>("ingestion");
  const [activeCaseId, setActiveCaseId] = useState<string>("souki_enclosures");
  const [activeRole, setActiveRole] = useState<string>("Admin");
  const [density, setDensity] = useState<"comfortable" | "compact">("comfortable");

  return (
    <div className={`flex w-full h-screen overflow-hidden bg-[#0b0f19] ${density === "compact" ? "text-[13px] leading-tight" : ""}`}>
      {/* Navigation Sidebar */}
      <Sidebar
        currentView={currentView}
        onSelectView={setCurrentView}
        activeRole={activeRole}
        onRoleChange={setActiveRole}
        density={density}
        onDensityChange={setDensity}
      />

      {/* Main Panel Content Area */}
      <main className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Unified Top Navigation & Diagnostics Header */}
        <UnifiedHeader
          currentView={currentView}
          onSelectView={setCurrentView}
          activeCaseId={activeCaseId}
          onSelectCase={setActiveCaseId}
          activeRole={activeRole}
          onRoleChange={setActiveRole}
        />

        {/* View Component Body */}
        <div className="flex-1 flex flex-col min-h-0 w-full overflow-hidden">
          <ErrorBoundary fallbackTitle={`${currentView.toUpperCase()} Module Error`}>
            {currentView === "ingestion" && <IngestionPipeline />}
            {currentView === "inspector" && <PdfInspector />}
            {currentView === "dashboard" && <CaseDashboard />}
            {currentView === "embedding" && <EmbeddingPipeline />}
            {currentView === "chat" && <RagChat />}
            {currentView === "diagnostics" && <SystemDiagnostics />}
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
