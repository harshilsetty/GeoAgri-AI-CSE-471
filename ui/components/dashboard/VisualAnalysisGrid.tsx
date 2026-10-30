"use client";

import Image from "next/image";
import { AnimatePresence, motion } from "framer-motion";
import { Info, Maximize2, Minus, Plus, RotateCcw, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { fadeSlideUp, modalBackdrop, modalPanel, staggerContainer, transitionBase, transitionFast } from "@/components/ui/motion";

export type VisualTile = {
  title: string;
  src: string;
  accent?: string;
  badge?: string;
  notice?: string;
  isObbMode?: boolean;
};

type VisualAnalysisGridProps = {
  tiles: VisualTile[];
  showVisuals: boolean;
  onToggleVisuals: () => void;
};

export default function VisualAnalysisGrid({ tiles, showVisuals, onToggleVisuals }: VisualAnalysisGridProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [touchStartX, setTouchStartX] = useState<number | null>(null);

  const expandedTile = expandedIndex !== null ? tiles[expandedIndex] ?? null : null;

  const canPan = zoom > 1;

  const transformStyle = useMemo(
    () => ({ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }),
    [pan.x, pan.y, zoom]
  );

  function openTile(index: number) {
    setExpandedIndex(index);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  const goNext = useCallback(() => {
    setExpandedIndex((prev) => {
      if (prev === null || tiles.length === 0) return prev;
      return (prev + 1) % tiles.length;
    });
  }, [tiles.length]);

  const goPrev = useCallback(() => {
    setExpandedIndex((prev) => {
      if (prev === null || tiles.length === 0) return prev;
      return (prev - 1 + tiles.length) % tiles.length;
    });
  }, [tiles.length]);

  useEffect(() => {
    if (expandedIndex === null) return;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setExpandedIndex(null);
        return;
      }
      if (event.key === "ArrowRight") {
        goNext();
        return;
      }
      if (event.key === "ArrowLeft") {
        goPrev();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [expandedIndex, goNext, goPrev]);

  return (
    <motion.section
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      variants={fadeSlideUp}
      transition={transitionBase}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <h3 className="inline-flex items-center gap-2 font-[var(--font-sora)] text-lg text-white">Visual Analysis Layers</h3>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="text-[11px] font-semibold uppercase tracking-[0.16em] text-dim transition hover:text-white"
          onClick={onToggleVisuals}
        >
          {showVisuals ? "Hide Visuals" : "Show Visuals"}
        </motion.button>
      </div>

      <div className={`overflow-hidden transition-all duration-500 ${showVisuals ? "max-h-[1600px] opacity-100" : "max-h-0 opacity-0"}`}>
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={staggerContainer}
          className="grid grid-cols-1 gap-3 pt-1 md:grid-cols-2 xl:grid-cols-4"
        >
          {tiles.map((tile, index) => (
            <motion.article
              key={tile.title}
              variants={{ hidden: { opacity: 0, y: 10 }, visible: { opacity: 1, y: 0 } }}
              whileHover={{ scale: 1.02, y: -4 }}
              className="group relative flex aspect-video flex-col justify-between overflow-hidden rounded-3xl border border-white/10 bg-slate-900/50 card-lift"
            >
              {tile.src ? (
                <Image src={tile.src} alt={tile.title} fill unoptimized className="object-cover transition duration-700 group-hover:scale-110" />
              ) : (
                <div className={`h-full w-full bg-gradient-to-br ${tile.accent || "from-slate-600/40 to-slate-900"} flex items-center justify-center p-4 text-center`}>
                  <p className="text-xs text-zinc-400">Run prediction to generate {tile.title.toLowerCase()}</p>
                </div>
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/25 to-black/40" />

              {/* Top Controls & Badges */}
              <div className="relative z-10 flex items-center justify-between p-3">
                {tile.badge ? (
                  <span className={`rounded-md px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider backdrop-blur ${
                    tile.isObbMode 
                      ? "border border-amber-400/40 bg-amber-500/25 text-amber-200" 
                      : tile.badge.includes("Heatmap") 
                        ? "border border-red-400/40 bg-red-500/25 text-red-200"
                        : "border border-cyan-400/30 bg-cyan-500/20 text-cyan-200"
                  }`}>
                    {tile.badge}
                  </span>
                ) : <span />}

                {tile.src ? (
                  <button
                    className="rounded-lg border border-white/15 bg-black/45 p-1.5 text-zinc-200 backdrop-blur transition hover:bg-black/60 hover:text-white"
                    onClick={() => openTile(index)}
                    aria-label={`Open ${tile.title} in full screen`}
                  >
                    <Maximize2 className="h-3.5 w-3.5" />
                  </button>
                ) : null}
              </div>

              {/* Bottom Title & Informative Notice */}
              <div className="relative z-10 p-3">
                <p className="inline-block rounded-lg bg-black/60 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-200 backdrop-blur">
                  {tile.title}
                </p>

                {tile.notice ? (
                  <p className="mt-1.5 flex items-start gap-1 rounded-md bg-black/75 p-1.5 text-[10px] leading-tight text-zinc-300 backdrop-blur border border-white/10">
                    <Info className="h-3 w-3 shrink-0 text-cyan-300 mt-0.5" />
                    <span>{tile.notice}</span>
                  </p>
                ) : null}
              </div>
            </motion.article>
          ))}
        </motion.div>
      </div>

      <AnimatePresence>
        {expandedTile?.src ? (
          <motion.div
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
            onClick={() => setExpandedIndex(null)}
            role="dialog"
            aria-modal="true"
            aria-label="Full screen visual"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={modalBackdrop}
            transition={transitionFast}
          >
            <motion.div
              className="relative h-[86vh] w-full max-w-6xl overflow-hidden rounded-2xl border border-white/10 bg-black flex flex-col justify-between"
              onClick={(e) => e.stopPropagation()}
              initial="hidden"
              animate="visible"
              exit="exit"
              variants={modalPanel}
              transition={transitionFast}
            >
              <div className="absolute inset-x-0 top-0 z-20 flex items-center justify-between bg-gradient-to-b from-black/85 to-transparent px-4 py-3">
                <div>
                  <p className="text-sm font-semibold text-white">{expandedTile.title}</p>
                  {expandedTile.notice ? (
                    <p className="text-[11px] text-amber-200/90">{expandedTile.notice}</p>
                  ) : null}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                    onClick={() => setZoom((prev) => Math.max(1, Number((prev - 0.2).toFixed(2))))}
                    aria-label="Zoom out"
                  >
                    <Minus className="h-4 w-4" />
                  </button>
                  <button
                    className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                    onClick={() => setZoom((prev) => Math.min(4, Number((prev + 0.2).toFixed(2))))}
                    aria-label="Zoom in"
                  >
                    <Plus className="h-4 w-4" />
                  </button>
                  <button
                    className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                    onClick={() => {
                      setZoom(1);
                      setPan({ x: 0, y: 0 });
                    }}
                    aria-label="Reset view"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </button>
                  <button
                    className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                    onClick={() => setExpandedIndex(null)}
                    aria-label="Close full screen"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div
                className={`relative h-full w-full ${canPan ? "cursor-grab active:cursor-grabbing" : "cursor-default"}`}
                onMouseDown={(e) => {
                  if (!canPan) return;
                  setDragging(true);
                  setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
                }}
                onMouseMove={(e) => {
                  if (!dragging || !canPan) return;
                  setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
                }}
                onMouseUp={() => setDragging(false)}
                onMouseLeave={() => setDragging(false)}
                onTouchStart={(e) => setTouchStartX(e.touches[0].clientX)}
                onTouchEnd={(e) => {
                  if (touchStartX === null) return;
                  const delta = e.changedTouches[0].clientX - touchStartX;
                  if (Math.abs(delta) > 45) {
                    if (delta < 0) goNext();
                    else goPrev();
                  }
                  setTouchStartX(null);
                }}
              >
                <div className="h-full w-full transition-transform duration-150" style={transformStyle}>
                  <Image src={expandedTile.src} alt={expandedTile.title} fill unoptimized className="object-contain" />
                </div>
              </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}
