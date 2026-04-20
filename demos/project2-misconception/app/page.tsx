"use client";

import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

/* ------------------------------------------------------------------ */
/*  Mock data                                                          */
/* ------------------------------------------------------------------ */

const COLORS = ["#06B6D4", "#D946EF", "#84CC16", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"];

interface Point { x: number; y: number; answer: string; question: string; score: number }
interface Cluster {
  id: number;
  label: string;
  color: string;
  cohesion: number;
  keywords: string[];
  points: Point[];
}

function generateClusters(): Cluster[] {
  const rng = (seed: number) => {
    let s = seed;
    return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
  };
  const r = rng(42);

  const defs = [
    { label: "Energy Confusion", keywords: ["energy", "force", "work", "power", "heat"], cx: -3, cy: 2 },
    { label: "Force Direction", keywords: ["force", "direction", "push", "pull", "gravity"], cx: 3, cy: 3 },
    { label: "Unit Confusion", keywords: ["unit", "measure", "kilogram", "newton", "joule"], cx: -2, cy: -3 },
    { label: "Process Reversal", keywords: ["reverse", "opposite", "backward", "wrong order"], cx: 4, cy: -2 },
    { label: "Scope Error", keywords: ["scope", "general", "specific", "broad", "narrow"], cx: 0, cy: -5 },
    { label: "Terminology Mix-up", keywords: ["term", "vocabulary", "definition", "meaning"], cx: -5, cy: 0 },
  ];

  const answers = [
    "Energy is the same as force because they both make things move",
    "Work and energy are the same thing just measured differently",
    "Force always goes in the direction of motion",
    "Gravity only pulls things down not sideways",
    "Kilograms and newtons measure the same thing",
    "The process happens in reverse order",
    "Heat and temperature are identical concepts",
    "Power is just another word for energy",
    "Mass and weight are exactly the same",
    "Friction always stops movement completely",
  ];

  return defs.map((d, i) => {
    const n = 20 + Math.floor(r() * 40);
    const points: Point[] = Array.from({ length: n }, () => ({
      x: d.cx + (r() - 0.5) * 3,
      y: d.cy + (r() - 0.5) * 3,
      answer: answers[Math.floor(r() * answers.length)],
      question: `q_${String(Math.floor(r() * 50) + 1).padStart(3, "0")}`,
      score: Math.floor(r() * 5) + 1,
    }));
    return { id: i, label: d.label, color: COLORS[i], cohesion: +(0.55 + r() * 0.35).toFixed(2), keywords: d.keywords, points };
  });
}

const CLUSTERS = generateClusters();

const keywordFreq = (() => {
  const freq: Record<string, number> = {};
  CLUSTERS.forEach((c) => c.keywords.forEach((k) => { freq[k] = (freq[k] || 0) + c.points.length; }));
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([keyword, count]) => ({ keyword, count }));
})();

/* ------------------------------------------------------------------ */
/*  Scatter plot (pure SVG — no Plotly dependency)                     */
/* ------------------------------------------------------------------ */

function ScatterPlot({
  clusters, selected, onSelect,
}: { clusters: Cluster[]; selected: number | null; onSelect: (id: number | null) => void }) {
  const allPoints = clusters.flatMap((c) => c.points.map((p) => ({ ...p, cid: c.id, color: c.color })));
  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  const minX = Math.min(...xs) - 1, maxX = Math.max(...xs) + 1;
  const minY = Math.min(...ys) - 1, maxY = Math.max(...ys) + 1;
  const w = 600, h = 400, pad = 20;
  const sx = (v: number) => pad + ((v - minX) / (maxX - minX)) * (w - 2 * pad);
  const sy = (v: number) => pad + ((maxY - v) / (maxY - minY)) * (h - 2 * pad);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ maxHeight: 420 }}>
      <rect width={w} height={h} rx={8} fill="#1E293B" />
      {allPoints.map((p, i) => (
        <circle
          key={i}
          cx={sx(p.x)}
          cy={sy(p.y)}
          r={4}
          fill={p.color}
          opacity={selected === null || selected === p.cid ? 0.85 : 0.12}
          className="cursor-pointer transition-opacity duration-200"
          onClick={() => onSelect(selected === p.cid ? null : p.cid)}
        >
          <title>{p.answer.slice(0, 60)}</title>
        </circle>
      ))}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Metric badge                                                       */
/* ------------------------------------------------------------------ */

function MetricBadge({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded bg-slate-700/60 px-2 py-0.5 text-[11px] font-mono text-slate-300">
      <span className="text-slate-500">{label}</span> {value}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Tabs                                                               */
/* ------------------------------------------------------------------ */

type Tab = "clusters" | "frequency" | "table";

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [tab, setTab] = useState<Tab>("clusters");
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);

  const cluster = selectedCluster !== null ? CLUSTERS[selectedCluster] : null;
  const totalAnswers = CLUSTERS.reduce((s, c) => s + c.points.length, 0);

  const tabs: { id: Tab; icon: string; label: string }[] = [
    { id: "clusters", icon: "🔍", label: "Clusters" },
    { id: "frequency", icon: "📈", label: "Frequency" },
    { id: "table", icon: "📋", label: "Table" },
  ];

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-14 bg-slate-900 border-r border-slate-800 flex flex-col items-center py-4 gap-1 shrink-0">
        <span className="text-lg mb-4">◉</span>
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            title={t.label}
            className={`w-10 h-10 rounded-lg flex items-center justify-center text-sm transition-colors ${
              tab === t.id ? "bg-cyan-600/20 text-cyan-400" : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {t.icon}
          </button>
        ))}
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="h-12 border-b border-slate-800 flex items-center px-5 gap-4 shrink-0">
          <h1 className="font-mono text-sm font-semibold text-slate-200">MisconceptionMiner</h1>
          <div className="flex gap-2 ml-auto">
            <MetricBadge label="Clusters" value={CLUSTERS.length} />
            <MetricBadge label="Answers" value={totalAnswers} />
            <MetricBadge label="Method" value="HDBSCAN" />
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-auto p-5">
          <AnimatePresence mode="wait">
            {/* ---- CLUSTERS TAB ---- */}
            {tab === "clusters" && (
              <motion.div key="clusters" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                {/* Scatter */}
                <div className="rounded-lg border border-slate-700 bg-slate-800 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">UMAP Cluster Visualization</h2>
                    <div className="flex gap-2">
                      <MetricBadge label="Silhouette" value="0.62" />
                      <MetricBadge label="NMI" value="0.71" />
                    </div>
                  </div>
                  <ScatterPlot clusters={CLUSTERS} selected={selectedCluster} onSelect={setSelectedCluster} />
                  {/* Legend */}
                  <div className="flex flex-wrap gap-3 mt-3">
                    {CLUSTERS.map((c) => (
                      <button
                        key={c.id}
                        onClick={() => setSelectedCluster(selectedCluster === c.id ? null : c.id)}
                        className={`flex items-center gap-1.5 text-[11px] transition-opacity ${
                          selectedCluster !== null && selectedCluster !== c.id ? "opacity-30" : ""
                        }`}
                      >
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color }} />
                        <span className="text-slate-400">{c.label}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Bottom: cluster list + detail */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                  {/* Cluster list */}
                  <div className="rounded-lg border border-slate-700 bg-slate-800 p-4 max-h-[400px] overflow-auto">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Cluster Summary</h3>
                    <div className="space-y-2">
                      {CLUSTERS.map((c) => (
                        <button
                          key={c.id}
                          onClick={() => setSelectedCluster(selectedCluster === c.id ? null : c.id)}
                          className={`w-full text-left rounded-lg px-3 py-2.5 transition-colors ${
                            selectedCluster === c.id ? "bg-slate-700" : "hover:bg-slate-700/50"
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                            <span className="text-sm text-slate-200 font-medium">#{c.id + 1} {c.label}</span>
                          </div>
                          <div className="ml-5 mt-1 text-[11px] text-slate-500">
                            {c.points.length} answers · cohesion {c.cohesion}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Detail panel */}
                  <div className="rounded-lg border border-slate-700 bg-slate-800 p-4 max-h-[400px] overflow-auto">
                    {cluster ? (
                      <>
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                          Selected Cluster: #{cluster.id + 1}
                        </h3>
                        <p className="text-lg font-semibold text-slate-100 mb-3" style={{ color: cluster.color }}>
                          {cluster.label}
                        </p>
                        <div className="flex flex-wrap gap-1.5 mb-4">
                          {cluster.keywords.map((k) => (
                            <span key={k} className="rounded bg-slate-700 px-2 py-0.5 text-[11px] text-cyan-300 font-mono">
                              {k}
                            </span>
                          ))}
                        </div>
                        <h4 className="text-[11px] text-slate-500 uppercase mb-2">Example Answers</h4>
                        <div className="space-y-2">
                          {cluster.points.slice(0, 5).map((p, i) => (
                            <div key={i} className="rounded bg-slate-900/60 p-3 text-sm text-slate-300">
                              &ldquo;{p.answer}&rdquo;
                              <div className="mt-1 text-[10px] text-slate-500">
                                Score: {p.score}/10 · {p.question}
                              </div>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-center h-full text-slate-600 text-sm">
                        Click a cluster to see details
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* ---- FREQUENCY TAB ---- */}
            {tab === "frequency" && (
              <motion.div key="frequency" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className="rounded-lg border border-slate-700 bg-slate-800 p-5">
                  <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                    Top Misconception Keywords
                  </h2>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={keywordFreq} layout="vertical" margin={{ left: 80 }}>
                      <XAxis type="number" stroke="#475569" tick={{ fill: "#94A3B8", fontSize: 11 }} />
                      <YAxis type="category" dataKey="keyword" stroke="#475569" tick={{ fill: "#94A3B8", fontSize: 12, fontFamily: "JetBrains Mono" }} width={70} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1E293B", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                        labelStyle={{ color: "#F1F5F9" }}
                      />
                      <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                        {keywordFreq.map((_, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Cluster size chart */}
                <div className="rounded-lg border border-slate-700 bg-slate-800 p-5 mt-5">
                  <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                    Cluster Sizes
                  </h2>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={CLUSTERS.map((c) => ({ name: c.label, count: c.points.length, color: c.color }))}>
                      <XAxis dataKey="name" stroke="#475569" tick={{ fill: "#94A3B8", fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                      <YAxis stroke="#475569" tick={{ fill: "#94A3B8", fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: "#1E293B", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {CLUSTERS.map((c, i) => (
                          <Cell key={i} fill={c.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}

            {/* ---- TABLE TAB ---- */}
            {tab === "table" && (
              <motion.div key="table" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className="rounded-lg border border-slate-700 bg-slate-800 overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-700 flex items-center gap-3">
                    <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">All Answers</h2>
                    <span className="text-[11px] text-slate-500">{totalAnswers} records</span>
                  </div>
                  <div className="overflow-auto max-h-[600px]">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-slate-800 border-b border-slate-700">
                        <tr className="text-[11px] text-slate-500 uppercase">
                          <th className="px-4 py-2 text-left w-16">#</th>
                          <th className="px-4 py-2 text-left">Student Answer</th>
                          <th className="px-4 py-2 text-left w-40">Cluster</th>
                          <th className="px-4 py-2 text-center w-20">Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {CLUSTERS.flatMap((c) =>
                          c.points.slice(0, 6).map((p, pi) => (
                            <tr key={`${c.id}-${pi}`} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                              <td className="px-4 py-2.5 text-slate-500 font-mono text-xs">{p.question}</td>
                              <td className="px-4 py-2.5 text-slate-300">&ldquo;{p.answer}&rdquo;</td>
                              <td className="px-4 py-2.5">
                                <span className="inline-flex items-center gap-1.5">
                                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />
                                  <span className="text-slate-400 text-xs">#{c.id + 1} {c.label}</span>
                                </span>
                              </td>
                              <td className="px-4 py-2.5 text-center text-slate-400 font-mono">{p.score}/10</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer */}
        <footer className="h-8 border-t border-slate-800 flex items-center px-5 text-[11px] text-slate-600 shrink-0">
          {CLUSTERS.length} clusters · {totalAnswers} answers · HDBSCAN (min_cluster=5) · UMAP 2D
        </footer>
      </div>
    </div>
  );
}
