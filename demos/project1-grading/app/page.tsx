"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Span {
  text: string;
  match: "matched" | "partial" | "missing" | "none";
}

interface GradeResult {
  score: number;
  confidence: number;
  label: "correct" | "partial" | "incorrect";
  similarity: { overall: number; semantic: number; lexical: number; keyConcept: number };
  explanation: string;
  concepts: { matched: string[]; missing: string[] };
  refSpans: Span[];
  stuSpans: Span[];
}

/* ------------------------------------------------------------------ */
/*  Example data                                                       */
/* ------------------------------------------------------------------ */

const EXAMPLES = [
  {
    question: "Explain the process of photosynthesis and why it is important for life on Earth.",
    referenceAnswer:
      "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. Plants absorb carbon dioxide and water, and using sunlight and chlorophyll, produce glucose and oxygen. This process is vital because it provides food for plants and oxygen for most living organisms.",
    studentAnswer:
      "Plants use sunlight to make food. They take in carbon dioxide and water and produce glucose. This is important because animals eat plants for energy.",
  },
  {
    question: "What is Newton's Third Law of Motion? Give an example.",
    referenceAnswer:
      "Newton's Third Law states that for every action there is an equal and opposite reaction. When one object exerts a force on a second object, the second object exerts an equal force in the opposite direction on the first. For example, when you push against a wall, the wall pushes back on you with equal force.",
    studentAnswer:
      "Newton's third law says every action has a reaction. Like when you push a wall it pushes you back.",
  },
  {
    question: "Describe the water cycle and its main stages.",
    referenceAnswer:
      "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth. Its main stages are evaporation, condensation, precipitation, and collection. Water evaporates from bodies of water, condenses into clouds, falls as precipitation, and collects in rivers, lakes, and oceans.",
    studentAnswer: "Water goes up as steam and comes back down as rain.",
  },
];

/* ------------------------------------------------------------------ */
/*  Small components                                                   */
/* ------------------------------------------------------------------ */

function Badge({ label }: { label: string }) {
  const cls =
    label === "correct"
      ? "bg-emerald-100 text-emerald-700"
      : label === "partial"
      ? "bg-amber-100 text-amber-700"
      : "bg-red-100 text-red-700";
  const icon = label === "correct" ? "✅" : label === "partial" ? "⚠️" : "❌";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-sm font-medium ${cls}`}>
      {icon} {label.charAt(0).toUpperCase() + label.slice(1)}
    </span>
  );
}

function SimilarityBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 text-xs text-gray-500 text-right">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-gray-100 overflow-hidden">
        <motion.div
          className="h-full rounded-full bg-amber-500"
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8 }}
        />
      </div>
      <span className="w-10 text-xs font-mono text-gray-600">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function AnnotatedText({ spans }: { spans: Span[] }) {
  return (
    <p className="leading-relaxed text-[15px]">
      {spans.map((s, i) => {
        if (s.match === "none") return <span key={i}>{s.text}</span>;
        const cls =
          s.match === "matched"
            ? "bg-emerald-100 text-emerald-800 rounded px-0.5"
            : s.match === "partial"
            ? "bg-amber-100 text-amber-800 rounded px-0.5"
            : "bg-red-100 text-red-800 rounded px-0.5 line-through decoration-red-300";
        return (
          <span key={i} className={cls}>
            {s.text}
          </span>
        );
      })}
    </p>
  );
}

/* ------------------------------------------------------------------ */
/*  Gauge (SVG)                                                        */
/* ------------------------------------------------------------------ */

function Gauge({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  const angle = pct * 180;
  const r = 60;
  const cx = 70;
  const cy = 70;
  const rad = (a: number) => ((a - 180) * Math.PI) / 180;
  const x = cx + r * Math.cos(rad(angle));
  const y = cy + r * Math.sin(rad(angle));

  return (
    <svg viewBox="0 0 140 85" className="w-36 mx-auto">
      <path d="M 10 70 A 60 60 0 0 1 130 70" fill="none" stroke="#E5E7EB" strokeWidth="10" strokeLinecap="round" />
      <motion.path
        d={`M 10 70 A 60 60 0 ${angle > 90 ? 1 : 0} 1 ${x} ${y}`}
        fill="none"
        stroke="#D97706"
        strokeWidth="10"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1 }}
      />
      <text x="70" y="72" textAnchor="middle" className="fill-gray-800 text-lg font-bold" fontSize="20">
        {(pct * 100).toFixed(0)}%
      </text>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [question, setQuestion] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [studentAnswer, setStudentAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GradeResult | null>(null);

  async function handleGrade() {
    if (!question.trim() || !referenceAnswer.trim() || !studentAnswer.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/grade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, referenceAnswer, studentAnswer }),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      alert("Grading failed — check console.");
    } finally {
      setLoading(false);
    }
  }

  function loadExample(idx: number) {
    const ex = EXAMPLES[idx];
    setQuestion(ex.question);
    setReferenceAnswer(ex.referenceAnswer);
    setStudentAnswer(ex.studentAnswer);
    setResult(null);
  }

  return (
    <div className="min-h-screen bg-cream">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white/80 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-2">
            <span className="text-2xl">📝</span>
            <h1 className="font-serif text-xl font-bold text-gray-900">ASAG Grader</h1>
          </div>
          <span className="text-xs text-gray-400">Teacher&apos;s Grading Dashboard</span>
        </div>
      </header>

      {/* Body */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* ---- LEFT: Input Panel ---- */}
          <div className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Question</label>
              <textarea
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Enter the question..."
                className="w-full rounded-lg border border-gray-200 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-amber-300 resize-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Reference Answer</label>
              <textarea
                rows={4}
                value={referenceAnswer}
                onChange={(e) => setReferenceAnswer(e.target.value)}
                placeholder="Enter the model / reference answer..."
                className="w-full rounded-lg border border-gray-200 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-amber-300 resize-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Student Answer</label>
              <textarea
                rows={4}
                value={studentAnswer}
                onChange={(e) => setStudentAnswer(e.target.value)}
                placeholder="Enter the student's answer..."
                className="w-full rounded-lg border border-gray-200 bg-white px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-amber-300 resize-none"
              />
            </div>

            <button
              onClick={handleGrade}
              disabled={loading || !question.trim() || !referenceAnswer.trim() || !studentAnswer.trim()}
              className="w-full rounded-lg bg-amber-600 hover:bg-amber-700 disabled:opacity-40 text-white font-semibold py-3 text-sm transition-colors flex items-center justify-center gap-2"
            >
              {loading ? (
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                "✨"
              )}
              {loading ? "Grading..." : "Grade Answer"}
            </button>

            {/* Examples */}
            <div>
              <p className="text-xs text-gray-400 mb-2">Quick examples:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => loadExample(i)}
                    className="rounded-full border border-gray-200 bg-white px-3 py-1 text-xs text-gray-600 hover:border-amber-400 hover:text-amber-700 transition-colors"
                  >
                    Example {i + 1}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ---- RIGHT: Results Panel ---- */}
          <div className="min-h-[400px]">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center h-full text-center text-gray-400 py-20"
                >
                  <span className="text-5xl mb-4">📊</span>
                  <p className="text-sm">Submit an answer to see grading results</p>
                </motion.div>
              )}

              {loading && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4"
                >
                  {[1, 2, 3].map((n) => (
                    <div key={n} className="rounded-xl border border-gray-100 bg-white p-6 animate-pulse">
                      <div className="h-4 bg-gray-100 rounded w-1/3 mb-3" />
                      <div className="h-8 bg-gray-100 rounded w-2/3" />
                    </div>
                  ))}
                </motion.div>
              )}

              {result && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="space-y-5"
                >
                  {/* Score card */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1 }}
                    className="rounded-xl border border-gray-100 bg-white p-6 shadow-sm"
                  >
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">Score</span>
                      <span className="text-4xl font-bold text-gray-900">
                        {result.score.toFixed(1)}
                        <span className="text-lg text-gray-300"> / 10</span>
                      </span>
                    </div>
                    <div className="mt-3 h-3 rounded-full bg-gray-100 overflow-hidden">
                      <motion.div
                        className={`h-full rounded-full ${
                          result.label === "correct"
                            ? "bg-emerald-500"
                            : result.label === "partial"
                            ? "bg-amber-500"
                            : "bg-red-500"
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${(result.score / 10) * 100}%` }}
                        transition={{ duration: 0.8 }}
                      />
                    </div>
                    <div className="mt-3 flex items-center justify-between">
                      <Badge label={result.label} />
                      <span className="text-sm text-gray-500">Confidence: {result.confidence}%</span>
                    </div>
                  </motion.div>

                  {/* Similarity */}
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.25 }}
                    className="rounded-xl border border-gray-100 bg-white p-6 shadow-sm"
                  >
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-4">
                      Similarity Analysis
                    </h3>
                    <Gauge value={result.similarity.overall} />
                    <div className="mt-4 space-y-2">
                      <SimilarityBar label="Semantic" value={result.similarity.semantic} />
                      <SimilarityBar label="Lexical" value={result.similarity.lexical} />
                      <SimilarityBar label="Key Concept" value={result.similarity.keyConcept} />
                    </div>
                  </motion.div>

                  {/* Explanation */}
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="rounded-xl border border-gray-100 bg-white p-6 shadow-sm"
                  >
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
                      Explanation
                    </h3>
                    <p className="text-sm text-gray-700 leading-relaxed mb-4">{result.explanation}</p>
                    <div className="flex flex-wrap gap-2">
                      {result.concepts.matched.map((c) => (
                        <span key={c} className="rounded-full bg-emerald-50 text-emerald-700 px-2.5 py-0.5 text-xs font-medium">
                          ✅ {c}
                        </span>
                      ))}
                      {result.concepts.missing.map((c) => (
                        <span key={c} className="rounded-full bg-red-50 text-red-700 px-2.5 py-0.5 text-xs font-medium">
                          ❌ {c}
                        </span>
                      ))}
                    </div>
                  </motion.div>

                  {/* Phrase alignment */}
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.55 }}
                    className="rounded-xl border border-gray-100 bg-white p-6 shadow-sm"
                  >
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-4">
                      Phrase Alignment
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <p className="text-[10px] text-gray-400 mb-1 uppercase">Reference Answer</p>
                        <AnnotatedText spans={result.refSpans} />
                      </div>
                      <div>
                        <p className="text-[10px] text-gray-400 mb-1 uppercase">Student Answer</p>
                        <AnnotatedText spans={result.stuSpans} />
                      </div>
                    </div>
                    <div className="mt-4 flex gap-4 text-[11px] text-gray-400">
                      <span>🟢 Matched</span>
                      <span>🟡 Partial</span>
                      <span>🔴 Missing</span>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}
