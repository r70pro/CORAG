"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Maximize2, Minimize2, RotateCcw, ArrowUp, ArrowDown, GripHorizontal } from "lucide-react";

interface ResizableBlockProps {
  id: string;
  title?: React.ReactNode;
  defaultHeight?: number; // Initial height in px (e.g. 400). If 0 or omitted, default is auto height.
  minHeight?: number;
  defaultWidth?: string | number; // e.g. "100%" or 600
  minWidth?: number;
  className?: string;
  headerActions?: React.ReactNode;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  canMove?: boolean;
  children: React.ReactNode;
}

export const ResizableBlock: React.FC<ResizableBlockProps> = ({
  id,
  title,
  defaultHeight = 0,
  minHeight = 120,
  defaultWidth = "100%",
  minWidth = 200,
  className = "",
  headerActions,
  onMoveUp,
  onMoveDown,
  canMove = false,
  children,
}) => {
  const [height, setHeight] = useState<number | null>(() => {
    if (!id || typeof window === "undefined") return defaultHeight > 0 ? defaultHeight : null;
    try {
      const saved = localStorage.getItem(`kirag_block_dim_${id}`);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (typeof parsed.height === "number") return parsed.height;
      }
    } catch (e) {
      console.error(`Error restoring height for block ${id}:`, e);
    }
    return defaultHeight > 0 ? defaultHeight : null;
  });

  const [width, setWidth] = useState<string | number>(() => {
    if (!id || typeof window === "undefined") return defaultWidth;
    try {
      const saved = localStorage.getItem(`kirag_block_dim_${id}`);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.width !== undefined) return parsed.width;
      }
    } catch (e) {
      console.error(`Error restoring width for block ${id}:`, e);
    }
    return defaultWidth;
  });

  const [isCollapsed, setIsCollapsed] = useState<boolean>(() => {
    if (!id || typeof window === "undefined") return false;
    try {
      const saved = localStorage.getItem(`kirag_block_dim_${id}`);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (typeof parsed.isCollapsed === "boolean") return parsed.isCollapsed;
      }
    } catch (e) {
      console.error(`Error restoring collapse state for block ${id}:`, e);
    }
    return false;
  });

  const [isMaximized, setIsMaximized] = useState<boolean>(false);
  const [isResizing, setIsResizing] = useState<"height" | "width" | "both" | null>(null);

  const blockRef = useRef<HTMLDivElement>(null);
  const startDragRef = useRef<{
    startX: number;
    startY: number;
    startWidth: number;
    startHeight: number;
  } | null>(null);

  // Save dimensions to localStorage
  const saveDimensions = useCallback(
    (h: number | null, w: string | number, collapsed: boolean) => {
      if (typeof window === "undefined" || !id) return;
      try {
        localStorage.setItem(
          `kirag_block_dim_${id}`,
          JSON.stringify({ height: h, width: w, isCollapsed: collapsed })
        );
      } catch (e) {
        console.error(`Error saving dimensions for block ${id}:`, e);
      }
    },
    [id]
  );

  const handleReset = () => {
    const nextH = defaultHeight > 0 ? defaultHeight : null;
    const nextW = defaultWidth;
    setHeight(nextH);
    setWidth(nextW);
    setIsCollapsed(false);
    setIsMaximized(false);
    saveDimensions(nextH, nextW, false);
  };

  const toggleCollapse = () => {
    const next = !isCollapsed;
    setIsCollapsed(next);
    saveDimensions(height, width, next);
  };

  const toggleMaximize = () => {
    setIsMaximized((prev) => !prev);
  };

  // Resize Handlers
  const startResizing = (type: "height" | "width" | "both", e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;

    if (!blockRef.current) return;
    const rect = blockRef.current.getBoundingClientRect();

    startDragRef.current = {
      startX: clientX,
      startY: clientY,
      startWidth: rect.width,
      startHeight: rect.height,
    };
    setIsResizing(type);
  };

  useEffect(() => {
    if (!isResizing) return;

    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (!startDragRef.current || !blockRef.current) return;
      const clientX = "touches" in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
      const clientY = "touches" in e ? e.touches[0].clientY : (e as MouseEvent).clientY;

      const deltaX = clientX - startDragRef.current.startX;
      const deltaY = clientY - startDragRef.current.startY;

      let newH = height;
      let newW = width;

      if (isResizing === "height" || isResizing === "both") {
        const calculatedH = Math.max(minHeight, startDragRef.current.startHeight + deltaY);
        newH = calculatedH;
        setHeight(calculatedH);
      }

      if (isResizing === "width" || isResizing === "both") {
        const calculatedW = Math.max(minWidth, startDragRef.current.startWidth + deltaX);
        newW = calculatedW;
        setWidth(calculatedW);
      }

      saveDimensions(newH, newW, isCollapsed);
    };

    const handleEnd = () => {
      setIsResizing(null);
      startDragRef.current = null;
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleEnd);
    window.addEventListener("touchmove", handleMove);
    window.addEventListener("touchend", handleEnd);

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleEnd);
      window.removeEventListener("touchmove", handleMove);
      window.removeEventListener("touchend", handleEnd);
    };
  }, [isResizing, height, width, minHeight, minWidth, isCollapsed, saveDimensions]);

  const widthStyle = typeof width === "number" ? `${width}px` : width;

  return (
    <div
      ref={blockRef}
      style={{
        width: isMaximized ? "100%" : widthStyle,
        height: isCollapsed ? "auto" : isMaximized ? "100%" : height ? `${height}px` : "auto",
        maxHeight: isMaximized ? "none" : undefined,
      }}
      className={`relative group flex flex-col transition-all duration-75 ${
        isMaximized ? "fixed inset-2 z-50 bg-slate-950 border border-indigo-500/80 shadow-2xl rounded-2xl" : ""
      } ${className}`}
    >
      {/* Resizing overlay indicator */}
      {isResizing && <div className="fixed inset-0 z-50 cursor-se-resize select-none" />}

      {/* Header bar with controls */}
      {(title || headerActions || canMove || defaultHeight > 0 || height !== null) && (
        <div className="flex items-center justify-between gap-2 border-b border-slate-800/80 pb-2 mb-3 shrink-0">
          <div className="flex items-center space-x-2 min-w-0">
            {canMove && (
              <div className="flex items-center space-x-1">
                {onMoveUp && (
                  <button
                    type="button"
                    onClick={onMoveUp}
                    className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
                    title="Move Block Up"
                  >
                    <ArrowUp className="w-3.5 h-3.5" />
                  </button>
                )}
                {onMoveDown && (
                  <button
                    type="button"
                    onClick={onMoveDown}
                    className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
                    title="Move Block Down"
                  >
                    <ArrowDown className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            )}
            {typeof title === "string" ? (
              <h3 className="text-sm font-bold text-slate-100 truncate">{title}</h3>
            ) : (
              title
            )}
          </div>

          <div className="flex items-center space-x-1 shrink-0">
            {headerActions}

            {(height !== null || typeof width === "number") && (
              <button
                type="button"
                onClick={handleReset}
                className="p-1 text-slate-400 hover:text-indigo-400 hover:bg-slate-900 rounded transition-colors"
                title="Reset Height & Width to Default"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}

            <button
              type="button"
              onClick={toggleCollapse}
              className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-900 rounded transition-colors"
              title={isCollapsed ? "Expand Block" : "Collapse Block"}
            >
              <GripHorizontal className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={toggleMaximize}
              className="p-1 text-slate-400 hover:text-slate-200 hover:bg-slate-900 rounded transition-colors"
              title={isMaximized ? "Restore Window Size" : "Maximize Window"}
            >
              {isMaximized ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      )}

      {/* Main Block Content */}
      {!isCollapsed && <div className="flex-1 min-h-0 min-w-0 flex flex-col overflow-auto">{children}</div>}

      {/* Drag Handles for Free Height & Width Adjustment */}
      {!isCollapsed && !isMaximized && (
        <>
          {/* Bottom Edge Drag Handle (Height) */}
          <div
            onMouseDown={(e) => startResizing("height", e)}
            onTouchStart={(e) => startResizing("height", e)}
            onDoubleClick={handleReset}
            className="absolute left-0 right-0 -bottom-1.5 h-3 cursor-row-resize flex items-center justify-center opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity z-20"
            title="Drag vertically to freely adjust height. Double-click to reset."
          >
            <div className="w-12 h-1 bg-indigo-500/80 rounded-full shadow-sm" />
          </div>

          {/* Right Edge Drag Handle (Width) */}
          <div
            onMouseDown={(e) => startResizing("width", e)}
            onTouchStart={(e) => startResizing("width", e)}
            onDoubleClick={handleReset}
            className="absolute top-0 bottom-0 -right-1.5 w-3 cursor-col-resize flex items-center justify-center opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity z-20"
            title="Drag horizontally to freely adjust width. Double-click to reset."
          >
            <div className="h-12 w-1 bg-indigo-500/80 rounded-full shadow-sm" />
          </div>

          {/* Bottom-Right Corner Handle (Height & Width) */}
          <div
            onMouseDown={(e) => startResizing("both", e)}
            onTouchStart={(e) => startResizing("both", e)}
            onDoubleClick={handleReset}
            className="absolute -right-1.5 -bottom-1.5 w-4 h-4 cursor-se-resize flex items-center justify-center opacity-40 group-hover:opacity-100 hover:opacity-100 transition-opacity z-30"
            title="Drag corner to freely adjust both width and height. Double-click to reset."
          >
            <div className="w-2.5 h-2.5 bg-indigo-400 rounded-sm shadow-md border border-indigo-200" />
          </div>
        </>
      )}
    </div>
  );
};
