"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface FeedbackResult {
  openingMessage: string;
  strengths: string[];
  weaknesses: string[];
  suggestions: string[];
  scores: { label: string; value: number }[];
}

type Tone = "friendly" | "academic" | "strict";

/* ------------------------------------------------------------------ */
/*  Examples                                                           */
/* ------------------------------------------------------------------ */

const EXAMPLES = [
  {
    question: "Explain how photosynthesis converts light energy into chemical energy stored in glucose.",
    studentAnswer: "Plants use sunlight to make food. They take in CO2 and water and produce glucose. Animals eat plants for energy.",
  },
  {
    question: "Describe the structure and function of DNA in living organisms.",
    studentAnswer: "DNA is like a twisted ladder. It has stuff called bases that make proteins.",
  },
  {
    question: "What causes the seasons on Earth?",
    studentAnswer: "The Earth is tilted on its axis at about 23.5 degrees. As it orbits the Sun, different hemispheres receive more direct sunlight at different times, causing seasonal temperature changes.",
  },
];

/* ------------------------------------------------------------------ */
/*  Typewriter hook                                                    */
/* ------------------------------------------------------------------ */

function useTypewriter(text: string, speed = 25) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  useEffect(() => {
    setDisplayed("");
    setDone(false);
    if (!text) return;
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) { clearInterval(interval); setDone(true); }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);
  return { displayed, done };
}

/* ------------------------------------------------------------------ */
/*  Tone selector                                                      */
/* ------------------------------------------------------------------ */

const TONES: { id: Tone; emoji: string; label: string }[] = [
  { id: "friendly", emoji: "😊", label: "Friendly" },
  { id: "academic", emoji: "📚", label: "Academic" },
  { id: "strict", emoji: "📏", label: "Strict" },
];

function ToneSelector({ selected, onChange }: { selected: Tone; onChange: (t: Tone) => void }) {
  return (
    <div className="flex items-center gap-1 bg-white/60 rounded-full p-1 backdrop-blur">
      {TONES.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`px-4 py-1.5 rounded-full text-sm font-semibold transition-all ${
            selected === t.id
              ? "bg-white text-violet-700 shadow-sm"
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          {t.emoji} {t.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Score bar                                                          */
/* ------------------------------------------------------------------ */

function ScoreBar({ label, value, delay }: { label: string; value: number; delay: number }) {
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-500 mb-1">
        <span>{label}</span>
        <span className="font-data">{value}%</span>
      </div>
      <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden">
        <motion.div
          className="h-full rounded-full bg-violet-500"
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.8, delay }}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [tone, setTone] = useState<Tone>("friendly");
  const [question, setQuestion] = useState("");
  const [studentAnswer, setStudentAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackResult | null>(null);
  const [submitted, setSubmitted] = useState<{ q: string; a: string } | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const { displayed: typedOpening, done: openingDone } = useTypewriter(
    feedback?.openingMessage ?? "", 20
  );

  async function handleSubmit() {
    if (!question.trim() || !studentAnswer.trim()) return;
    setLoading(true);
    setFeedback(null);
    setSubmitted({ q: question, a: studentAnswer });
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, studentAnswer, tone }),
      });
      const data = await res.json();
      setFeedback(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch {
      alert("Feedback generation failed");
    } finally {
      setLoading(false);
    }
  }

  function loadExample(idx: number) {
    setQuestion(EXAMPLES[idx].question);
    setStudentAnswer(EXAMPLES[idx].studentAnswer);
    setFeedback(null);
    setSubmitted(null);
  }

  return (
    <div className="min-h-screen">
      <div className="max-w-2xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-extrabold text-indigo-950 mb-1">🎓 StudyBuddy</h1>
          <p className="text-sm text-slate-500 mb-4">Your AI Learning Assistant</p>
          <ToneSelector selected={tone} onChange={setTone} />
        </div>

        {/* Conversation area */}
        <div className="space-y-4 mb-6">
          {/* Question bubble */}
          {submitted && (
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex justify-start">
              <div className="max-w-[85%] rounded-2xl rounded-bl-md bg-white border border-slate-200 px-5 py-3 shadow-sm">
                <p className="text-[10px] text-slate-400 uppercase font-semibold mb-1">📝 Question</p>
                <p className="text-sm text-slate-700">{submitted.q}</p>
              </div>
            </motion.div>
          )}

          {/* Student answer bubble */}
          {submitted && (
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.15 }} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-violet-600 to-violet-700 px-5 py-3 shadow-sm">
                <p className="text-[10px] text-violet-200 uppercase font-semibold mb-1">✏️ Your Answer</p>
                <p className="text-sm text-white">{submitted.a}</p>
              </div>
            </motion.div>
          )}

          {/* Loading */}
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="rounded-2xl bg-white border border-slate-200 px-5 py-4 shadow-sm">
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-violet-400 border-t-transparent" />
                  StudyBuddy is thinking...
                </div>
              </div>
            </motion.div>
          )}

          {/* Feedback card */}
          <AnimatePresence>
            {feedback && (
              <motion.div
                ref={resultRef}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="rounded-2xl bg-white border border-slate-100 shadow-lg p-6 space-y-4"
              >
                {/* Opening message */}
                <p className="text-sm text-slate-700 leading-relaxed">
                  {typedOpening}
                  {!openingDone && <span className="animate-pulse">▊</span>}
                </p>

                {/* Strengths */}
                <motion.div
                  initial={{ opacity: 0, x: -15 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                  className="rounded-xl bg-emerald-50 border border-emerald-200 p-4"
                >
                  <h3 className="text-sm font-bold text-emerald-700 mb-2">💪 Strengths</h3>
                  <ul className="space-y-1.5">
                    {feedback.strengths.map((s, i) => (
                      <li key={i} className="flex gap-2 text-sm text-emerald-800">
                        <span className="text-emerald-500 shrink-0">✅</span> {s}
                      </li>
                    ))}
                  </ul>
                </motion.div>

                {/* Weaknesses */}
                <motion.div
                  initial={{ opacity: 0, x: -15 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                  className="rounded-xl bg-amber-50 border border-amber-200 p-4"
                >
                  <h3 className="text-sm font-bold text-amber-700 mb-2">🔍 Areas to Improve</h3>
                  <ul className="space-y-1.5">
                    {feedback.weaknesses.map((w, i) => (
                      <li key={i} className="flex gap-2 text-sm text-amber-800">
                        <span className="text-amber-500 shrink-0">⚠️</span> {w}
                      </li>
                    ))}
                  </ul>
                </motion.div>

                {/* Suggestions */}
                <motion.div
                  initial={{ opacity: 0, x: -15 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 }}
                  className="rounded-xl bg-blue-50 border border-blue-200 p-4"
                >
                  <h3 className="text-sm font-bold text-blue-700 mb-2">💡 Suggestions</h3>
                  <ol className="space-y-1.5 list-decimal list-inside">
                    {feedback.suggestions.map((s, i) => (
                      <li key={i} className="text-sm text-blue-800">{s}</li>
                    ))}
                  </ol>
                </motion.div>

                {/* Scores */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.9 }}
                  className="rounded-xl bg-slate-50 border border-slate-200 p-4"
                >
                  <h3 className="text-sm font-bold text-slate-700 mb-3">📊 Your Score</h3>
                  <div className="space-y-3">
                    {feedback.scores.map((s, i) => (
                      <ScoreBar key={s.label} label={s.label} value={s.value} delay={1.0 + i * 0.1} />
                    ))}
                  </div>
                </motion.div>

                {/* Helpfulness */}
                <div className="flex items-center justify-center gap-4 pt-2 text-sm text-slate-500">
                  <span>Was this helpful?</span>
                  <button className="px-3 py-1 rounded-full border border-slate-200 hover:bg-emerald-50 hover:border-emerald-300 transition">
                    👍 Yes
                  </button>
                  <button className="px-3 py-1 rounded-full border border-slate-200 hover:bg-red-50 hover:border-red-300 transition">
                    👎 No
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Input area */}
        <div className="rounded-2xl bg-white border border-slate-200 shadow-sm p-5 space-y-3">
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">📝 Question</label>
            <textarea
              rows={2}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter the question..."
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-300 resize-none"
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 mb-1">✏️ Your Answer</label>
            <textarea
              rows={3}
              value={studentAnswer}
              onChange={(e) => setStudentAnswer(e.target.value)}
              placeholder="Type your answer here..."
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-300 resize-none"
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              {EXAMPLES.map((_, i) => (
                <button
                  key={i}
                  onClick={() => loadExample(i)}
                  className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-500 hover:border-violet-400 hover:text-violet-600 transition"
                >
                  Example {i + 1}
                </button>
              ))}
            </div>
            <button
              onClick={handleSubmit}
              disabled={loading || !question.trim() || !studentAnswer.trim()}
              className="rounded-full bg-violet-600 hover:bg-violet-700 disabled:opacity-40 text-white font-semibold px-6 py-2.5 text-sm transition-colors flex items-center gap-2"
            >
              {loading ? (
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                "🚀"
              )}
              {loading ? "Generating..." : "Get Feedback"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
