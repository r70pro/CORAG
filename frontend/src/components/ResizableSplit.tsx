"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";

interface ResizableSplitProps {
  direction?: "horizontal" | "vertical";
  initialSizes?: number[];
  minSizes?: number[];
  storageKey?: string;
  className?: string;
  children: React.ReactNode[];
}

export const ResizableSplit: React.FC<ResizableSplitProps> = ({
  direction = "horizontal",
  initialSizes,
  minSizes,
  storageKey,
  className = "",
  children,
}) => {
  const childArray = React.Children.toArray(children);
  const count = childArray.length;

  // Helper to compare array contents for shallow equality
  const areArraysEqual = (a?: number[], b?: number[]) => {
    if (a === b) return true;
    if (!a || !b) return false;
    if (a.length !== b.length) return false;
    return a.every((val, idx) => val === b[idx]);
  };

  // Memoize fallback defaults
  const fallbackSizes = React.useMemo(() => {
    if (initialSizes && initialSizes.length === count) return initialSizes;
    return new Array(count).fill(100 / count);
  }, [count, initialSizes]);

  const defaultMinSizes = React.useMemo(() => {
    if (minSizes && minSizes.length === count) return minSizes;
    return new Array(count).fill(0); // Allow full 0% collapse by default for max flexibility
  }, [count, minSizes]);

  const [sizes, setSizes] = useState<number[]>(fallbackSizes);

  const [prevFallback, setPrevFallback] = useState<number[]>(fallbackSizes);
  if (!areArraysEqual(prevFallback, fallbackSizes)) {
    setPrevFallback(fallbackSizes);
    setSizes(fallbackSizes);
  }

  // Restore stored split sizes post-mount to prevent SSR hydration mismatch
  useEffect(() => {
    if (!storageKey || typeof window === "undefined") return;
    const animId = requestAnimationFrame(() => {
      try {
        const saved = localStorage.getItem(`kirag_split_${storageKey}`);
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed) && parsed.length === count) {
            setSizes(parsed);
          }
        }
      } catch (e) {
        console.error("Error reading stored split sizes:", e);
      }
    });
    return () => cancelAnimationFrame(animId);
  }, [storageKey, count]);

  const [isDragging, setIsDragging] = useState<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    index: number;
    startPos: number;
    startSizes: number[];
  } | null>(null);

  const saveSizes = useCallback(
    (newSizes: number[]) => {
      if (storageKey && typeof window !== "undefined") {
        try {
          localStorage.setItem(`kirag_split_${storageKey}`, JSON.stringify(newSizes));
        } catch (e) {
          console.error("Error saving split sizes:", e);
        }
      }
    },
    [storageKey]
  );

  const handleMouseDown = (index: number, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const startPos = direction === "horizontal" ? e.clientX : e.clientY;
    dragRef.current = {
      index,
      startPos,
      startSizes: [...sizes],
    };
    setIsDragging(index);
  };

  const handleTouchStart = (index: number, e: React.TouchEvent) => {
    if (e.touches.length !== 1) return;
    const touch = e.touches[0];
    const startPos = direction === "horizontal" ? touch.clientX : touch.clientY;
    dragRef.current = {
      index,
      startPos,
      startSizes: [...sizes],
    };
    setIsDragging(index);
  };

  const handleDoubleClick = () => {
    // Reset to default fallback sizes on double click
    const nextSizes = [...fallbackSizes];
    setSizes(nextSizes);
    saveSizes(nextSizes);
  };


  const onMove = useCallback(
    (clientX: number, clientY: number) => {
      if (!dragRef.current || !containerRef.current) return;

      const { index, startPos, startSizes } = dragRef.current;
      const containerRect = containerRef.current.getBoundingClientRect();
      const containerSize =
        direction === "horizontal" ? containerRect.width : containerRect.height;

      if (containerSize <= 0) return;

      const currentPos = direction === "horizontal" ? clientX : clientY;
      const deltaPx = currentPos - startPos;
      const deltaPct = (deltaPx / containerSize) * 100;

      const minLeft = defaultMinSizes[index] ?? 0;
      const minRight = defaultMinSizes[index + 1] ?? 0;

      let newLeftSize = startSizes[index] + deltaPct;
      let newRightSize = startSizes[index + 1] - deltaPct;

      if (newLeftSize < minLeft) {
        newLeftSize = minLeft;
        newRightSize = startSizes[index] + startSizes[index + 1] - minLeft;
      } else if (newRightSize < minRight) {
        newRightSize = minRight;
        newLeftSize = startSizes[index] + startSizes[index + 1] - minRight;
      }

      newLeftSize = Math.max(minLeft, Math.min(100, newLeftSize));
      newRightSize = Math.max(minRight, Math.min(100, newRightSize));

      const nextSizes = [...startSizes];
      nextSizes[index] = newLeftSize;
      nextSizes[index + 1] = newRightSize;

      setSizes(nextSizes);
      saveSizes(nextSizes);
    },
    [direction, defaultMinSizes, saveSizes]
  );

  useEffect(() => {
    if (isDragging === null) return;

    const handleMouseMove = (e: MouseEvent) => {
      onMove(e.clientX, e.clientY);
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        onMove(e.touches[0].clientX, e.touches[0].clientY);
      }
    };

    const handleEnd = () => {
      setIsDragging(null);
      dragRef.current = null;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleEnd);
    window.addEventListener("touchmove", handleTouchMove);
    window.addEventListener("touchend", handleEnd);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleEnd);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleEnd);
    };
  }, [isDragging, onMove]);

  const isHoriz = direction === "horizontal";

  return (
    <div
      ref={containerRef}
      className={`relative flex w-full h-full min-h-0 min-w-0 ${
        isHoriz ? "flex-row" : "flex-col"
      } ${className}`}
    >
      {/* Global overlay during drag to prevent iframe / selection capture */}
      {isDragging !== null && (
        <div className="fixed inset-0 z-50 select-none cursor-col-resize cursor-row-resize" />
      )}

      {childArray.map((child, i) => {
        const sizePct = sizes[i] ?? 100 / count;
        const isLast = i === count - 1;

        return (
          <React.Fragment key={i}>
            {/* Child Panel Container */}
            <div
              style={{
                flexBasis: `${sizePct}%`,
                flexGrow: 0,
                flexShrink: 0,
                [isHoriz ? "width" : "height"]: `${sizePct}%`,
              }}
              className="relative min-w-0 min-h-0 flex flex-col overflow-hidden transition-none"
            >
              {child}
            </div>

            {/* Drag Handle Divider */}
            {!isLast && (
              <div
                role="separator"
                aria-orientation={direction}
                title="Drag to resize. Double-click to reset panel sizes."
                onMouseDown={(e) => handleMouseDown(i, e)}
                onTouchStart={(e) => handleTouchStart(i, e)}
                onDoubleClick={handleDoubleClick}
                className={`group relative z-20 flex items-center justify-center shrink-0 transition-colors select-none ${
                  isHoriz
                    ? "w-2.5 -mx-1 cursor-col-resize hover:w-3"
                    : "h-2.5 -my-1 cursor-row-resize hover:h-3"
                } ${
                  isDragging === i
                    ? "bg-indigo-500 shadow-lg shadow-indigo-500/30"
                    : "bg-slate-800/40 hover:bg-indigo-500/70"
                }`}
              >
                {/* Visual Handle Bar / Indicator */}
                <div
                  className={`rounded-full transition-all ${
                    isDragging === i
                      ? "bg-indigo-300"
                      : "bg-slate-600 group-hover:bg-indigo-300"
                  } ${isHoriz ? "w-1 h-8" : "h-1 w-8"}`}
                />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
