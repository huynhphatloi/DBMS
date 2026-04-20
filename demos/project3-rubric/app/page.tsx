"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
} from "recharts";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Criterion { name: string; weight: number; description: string }

interface CriterionScore {
  criterion: string; weight: number; score: number; maxScore: number;
  matchedTerms: string[]; explanation: string;
}

interface EvalResult {
  scores: CriterionScore[]; finalScore: number; maxScore: number; grade: string;
}

/* ------------------------------------------------------------------ */
/*  Default data                                                       */
/* ------------------------------------------------------------------ */

const DEFAULT_CRITERIA: Criterion[] = [
  { name: "Core Process", weight: 30, description: "Mentions light energy to chemical energy conversion in chloroplasts" },
  { name: "Key Components", weight: 25, description: "Identifies chlorophyll carbon dioxide water glucose oxygen" },
  { name: "Importance", weight: 25, description: "Explains role in food chains oxygen supply and ecosystem" },
  { name: "Scientific Language", weight: 20, description: "Uses correct terminology and clear scientific explanations" },
];

const COLORS = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626"];

/* ------------------------------------------------------------------ */
/*  Steps                                                              */
/* ------------------------------------------------------------------ */

type Step = 1 | 2 | 3;

function StepIndicator({ current }: { current: Step }) {
  const steps = [
    { n: 1, label: "Setup" },
    { n: 2, label: "Grading" },
    { n: 3, label: "Results" },
  ];
  return (
    <div className="flex items-center gap-2">
      {steps.map((s, i) => (
        <div key={s.n} className="flex items-center gap-2">
          <div
            className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
              current >= s.n ? "bg-blue-600 text-white" : "bg-slate-200 text-slate-400"
            }`}
          >
            {current > s.n ? "✓" : s.n}
          </div>
          <span className={`text-sm ${current >= s.n ? "text-slate-700 font-medium" : "text-slate-400"}`}>
            {s.label}
          </span>
          {i < steps.length - 1 && (
            <div className={`w-10 h-0.5 ${current > s.n ? "bg-blue-600" : "bg-slate-200"}`} />
          )}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [step, setStep] = useState<Step>(1);
  const [question, setQuestion] = useState(
    "Explain the process of photosynthesis and its importance for life on Earth."
  );
  const [criteria, setCriteria] = useState<Criterion[]>(DEFAULT_CRITERIA);
  const [studentAnswer, setStudentAnswer] = useState(
    "Plants use sunlight to make food. They take in carbon dioxide and water and produce glucose. This is important because animals eat plants."
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResult | null>(null);

  const totalWeight = criteria.reduce((s, c) => s + c.weight, 0);
  const weightValid = Math.abs(totalWeight - 100) < 0.5;

  function updateCriterion(idx: number, field: keyof Criterion, value: string | number) {
    setCriteria((prev) => prev.map((c, i) => (i === idx ? { ...c, [field]: value } : c)));
  }

  function addCriterion() {
    setCriteria((prev) => [...prev, { name: "New Criterion", weight: 0, description: "" }]);
  }

  function removeCriterion(idx: number) {
    if (criteria.length <= 2) return;
    setCriteria((prev) => prev.filter((_, i) => i !== idx));
  }

  async function handleEvaluate() {
    if (!weightValid) return;
    setLoading(true);
    setStep(2);
    try {
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, criteria, studentAnswer }),
      });
      const data = await res.json();
      setResult(data);
      setStep(3);
    } catch {
      alert("Evaluation failed");
      setStep(1);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-30">
        <div className="max-w-5xl mx-auto flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-2">
            <span className="text-xl">🎓</span>
            <h1 className="text-lg font-bold text-slate-900">RubricGrader</h1>
          </div>
          <StepIndicator current={step} />
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          {/* ---- STEP 1: Setup ---- */}
          {(step === 1 || step === 2) && !result && (
            <motion.div key="setup" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-6">
              {/* Question */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Question</label>
                <textarea
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300 resize-none"
                />
              </div>

              {/* Rubric table */}
              <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
                  <h3 className="text-sm font-semibold text-slate-700">Rubric Criteria</h3>
                  <button onClick={addCriterion} className="text-sm text-blue-600 hover:text-blue-800">+ Add</button>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-slate-50 text-[11px] text-slate-500 uppercase">
                      <th className="px-4 py-2 text-left">Criterion</th>
                      <th className="px-4 py-2 text-center w-28">Weight %</th>
                      <th className="px-4 py-2 text-left">Description</th>
                      <th className="px-4 py-2 w-10" />
                    </tr>
                  </thead>
                  <tbody>
                    {criteria.map((c, i) => (
                      <tr key={i} className="border-b last:border-0">
                        <td className="px-4 py-2">
                          <input
                            value={c.name}
                            onChange={(e) => updateCriterion(i, "name", e.target.value)}
                            className="w-full border-0 bg-transparent font-medium focus:outline-none focus:ring-1 focus:ring-blue-200 rounded px-1"
                          />
                        </td>
                        <td className="px-4 py-2 text-center">
                          <input
                            type="number"
                            min={0}
                            max={100}
                            value={c.weight}
                            onChange={(e) => updateCriterion(i, "weight", Number(e.target.value))}
                            className="w-16 text-center border border-slate-200 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-300"
                          />
                        </td>
                        <td className="px-4 py-2">
                          <input
                            value={c.description}
                            onChange={(e) => updateCriterion(i, "description", e.target.value)}
                            className="w-full border-0 bg-transparent text-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-200 rounded px-1"
                          />
                        </td>
                        <td className="px-4 py-2">
                          {criteria.length > 2 && (
                            <button onClick={() => removeCriterion(i)} className="text-slate-400 hover:text-red-500">×</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {/* Weight bar */}
                <div className="px-4 py-3 border-t bg-slate-50">
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-slate-500">Total Weight</span>
                    <span className={weightValid ? "text-emerald-600 font-medium" : "text-red-600 font-medium"}>
                      {totalWeight}% {weightValid ? "✓" : "(must = 100%)"}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        weightValid ? "bg-emerald-500" : totalWeight > 100 ? "bg-red-500" : "bg-amber-500"
                      }`}
                      style={{ width: `${Math.min(totalWeight, 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Student answer */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Student Answer</label>
                <textarea
                  rows={4}
                  value={studentAnswer}
                  onChange={(e) => setStudentAnswer(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300 resize-none"
                />
              </div>

              <button
                onClick={handleEvaluate}
                disabled={loading || !weightValid}
                className="w-full rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-semibold py-3 text-sm transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                ) : (
                  "▶"
                )}
                {loading ? "Evaluating..." : "Evaluate with Rubric"}
              </button>
            </motion.div>
          )}

          {/* ---- STEP 3: Results ---- */}
          {step === 3 && result && (
            <motion.div key="results" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left: scores */}
                <div className="space-y-5">
                  {/* Final score */}
                  <div className="rounded-xl border border-slate-200 bg-white p-6">
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">Final Score</span>
                      <div>
                        <span className="text-4xl font-bold text-slate-900">{result.finalScore.toFixed(1)}</span>
                        <span className="text-lg text-slate-300"> / {result.maxScore}</span>
                      </div>
                    </div>
                    <div className="mt-3 h-3 rounded-full bg-slate-100 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-blue-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${(result.finalScore / result.maxScore) * 100}%` }}
                        transition={{ duration: 0.8 }}
                      />
                    </div>
                    <div className="mt-2 text-right">
                      <span className="inline-block rounded-full bg-blue-100 text-blue-700 px-3 py-0.5 text-sm font-bold">
                        Grade: {result.grade}
                      </span>
                    </div>
                  </div>

                  {/* Per-criterion bars */}
                  <div className="rounded-xl border border-slate-200 bg-white p-6 space-y-4">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Per-Criterion Scores</h3>
                    {result.scores.map((s, i) => (
                      <div key={i}>
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="font-medium text-slate-700">{s.criterion} ({s.weight}%)</span>
                          <span className="font-mono text-slate-500">{s.score.toFixed(1)}/{s.maxScore}</span>
                        </div>
                        <div className="h-2.5 rounded-full bg-slate-100 overflow-hidden">
                          <motion.div
                            className="h-full rounded-full"
                            style={{ backgroundColor: COLORS[i % COLORS.length] }}
                            initial={{ width: 0 }}
                            animate={{ width: `${(s.score / s.maxScore) * 100}%` }}
                            transition={{ duration: 0.6, delay: i * 0.1 }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Right: charts */}
                <div className="space-y-5">
                  {/* Radar */}
                  <div className="rounded-xl border border-slate-200 bg-white p-6">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Radar Chart</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <RadarChart data={result.scores.map((s) => ({ subject: s.criterion, score: s.score, max: s.maxScore }))}>
                        <PolarGrid stroke="#E2E8F0" />
                        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "#64748B" }} />
                        <PolarRadiusAxis angle={90} domain={[0, 10]} tick={{ fontSize: 10, fill: "#94A3B8" }} />
                        <Radar name="Score" dataKey="score" stroke="#2563EB" fill="#2563EB" fillOpacity={0.2} strokeWidth={2} />
                        <Radar name="Max" dataKey="max" stroke="#E2E8F0" fill="none" strokeDasharray="4 4" />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Stacked bar */}
                  <div className="rounded-xl border border-slate-200 bg-white p-6">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Weighted Contributions</h3>
                    <ResponsiveContainer width="100%" height={80}>
                      <BarChart
                        layout="vertical"
                        data={[result.scores.reduce((acc, s, i) => {
                          acc[s.criterion] = Math.round((s.score * s.weight / totalWeight) * 100) / 100;
                          return acc;
                        }, {} as Record<string, number>)]}
                        stackOffset="expand"
                      >
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey={() => ""} hide />
                        <Tooltip contentStyle={{ fontSize: 12 }} />
                        {result.scores.map((s, i) => (
                          <Bar key={s.criterion} dataKey={s.criterion} stackId="a" fill={COLORS[i % COLORS.length]} radius={i === 0 ? [4, 0, 0, 4] : i === result.scores.length - 1 ? [0, 4, 4, 0] : 0} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="flex flex-wrap gap-3 mt-2">
                      {result.scores.map((s, i) => (
                        <span key={i} className="flex items-center gap-1 text-[11px] text-slate-500">
                          <span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                          {s.criterion}: {((s.score * s.weight) / totalWeight).toFixed(2)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Explainability */}
              <div className="rounded-xl border border-slate-200 bg-white p-6">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-4">
                  Explainability: Answer ↔ Rubric Alignment
                </h3>
                <div className="space-y-3">
                  {result.scores.map((s, i) => (
                    <div key={i} className="flex gap-3">
                      <span
                        className="inline-flex items-center justify-center w-6 h-6 rounded-full text-[10px] text-white font-bold shrink-0 mt-0.5"
                        style={{ backgroundColor: COLORS[i % COLORS.length] }}
                      >
                        {i + 1}
                      </span>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm text-slate-700">{s.criterion}</span>
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            s.score >= 7 ? "bg-emerald-100 text-emerald-700" :
                            s.score >= 4 ? "bg-amber-100 text-amber-700" :
                            "bg-red-100 text-red-700"
                          }`}>
                            {s.score >= 7 ? "strong" : s.score >= 4 ? "partial" : "weak"}
                          </span>
                        </div>
                        <p className="text-sm text-slate-500 mt-0.5">{s.explanation}</p>
                        {s.matchedTerms.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {s.matchedTerms.map((t) => (
                              <span key={t} className="rounded bg-blue-50 text-blue-600 px-1.5 py-0.5 text-[10px] font-mono">{t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={() => { setStep(1); setResult(null); }}
                  className="rounded-lg border border-slate-200 bg-white px-5 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
                >
                  ← Back to Setup
                </button>
                <button
                  onClick={() => { setStudentAnswer(""); setStep(1); setResult(null); }}
                  className="rounded-lg bg-blue-600 hover:bg-blue-700 px-5 py-2.5 text-sm font-medium text-white"
                >
                  Grade Another
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
