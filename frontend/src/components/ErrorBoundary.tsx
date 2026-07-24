"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error caught by ErrorBoundary:", error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="m-6 p-6 rounded-xl bg-red-950/30 border border-red-800/50 text-red-200 shadow-xl">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2.5 rounded-lg bg-red-900/50 text-red-400">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-red-100">
                {this.props.fallbackTitle || "Component Error Detected"}
              </h3>
              <p className="text-sm text-red-300/80">
                An unexpected runtime error occurred in this view module. The rest of the application remains active.
              </p>
            </div>
          </div>

          {this.state.error && (
            <div className="mb-4 p-3 rounded-lg bg-black/40 border border-red-900/30 font-mono text-xs text-red-300 overflow-x-auto">
              {this.state.error.toString()}
            </div>
          )}

          <button
            onClick={this.handleReset}
            className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg bg-red-800/60 hover:bg-red-700/80 text-white transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-red-500"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Try Reloading Component
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
