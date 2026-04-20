# Project 1 — Automatic Short Answer Grading

## UI Concept

**"Teacher's Grading Desk"** — A warm, professional grading dashboard that feels like a digital version of a teacher's workspace. Think Google Classroom meets Grammarly. The design uses a split-panel layout: input on the left, results on the right. The color palette is warm neutrals (cream, soft blue, amber accents) with a serif heading font to evoke an academic feel.

Visual identity: Clean card-based layout, subtle paper textures on input areas, a prominent "Grade" action button in amber/gold, and smooth reveal animations for results.

---

## Main Screens

### Screen 1: Grading Workspace (Primary — Single Page App)

```
┌─────────────────────────────────────────────────────────────────────┐
│  [Logo] ASAG Grader            [History ▾]  [Settings ⚙]  [?]     │
├────────────────────────────────┬────────────────────────────────────┤
│                                │                                    │
│  📝 INPUT PANEL                │  📊 RESULTS PANEL                  │
│                                │                                    │
│  ┌──────────────────────────┐  │  (empty state: illustration +     │
│  │ Question                 │  │   "Submit an answer to begin")    │
│  │ [textarea]               │  │                                    │
│  └──────────────────────────┘  │  ── after grading ──              │
│                                │                                    │
│  ┌──────────────────────────┐  │  ┌────────────────────────────┐   │
│  │ Reference Answer         │  │  │  SCORE        8.5 / 10     │   │
│  │ [textarea]               │  │  │  ████████████░░  (85%)     │   │
│  └──────────────────────────┘  │  │  Confidence: 92%           │   │
│                                │  │  Label: ✅ Correct          │   │
│  ┌──────────────────────────┐  │  └────────────────────────────┘   │
│  │ Student Answer           │  │                                    │
│  │ [textarea — highlighted] │  │  ┌────────────────────────────┐   │
│  └──────────────────────────┘  │  │  SIMILARITY ANALYSIS       │   │
│                                │  │  ┌─────────────────────┐   │   │
│  [Classification ▾] [0-10 ▾]  │  │  │  Gauge: 0.87        │   │   │
│                                │  │  └─────────────────────┘   │   │
│  ┌──────────────────────────┐  │  │  Semantic: 0.91            │   │
│  │                          │  │  │  Lexical:  0.72            │   │
│  │   [ ✨ GRADE ANSWER ]    │  │  │  Key Concept: 0.85         │   │
│  │                          │  │  └────────────────────────────┘   │
│  └──────────────────────────┘  │                                    │
│                                │  ┌────────────────────────────┐   │
│  ── Quick Examples ──          │  │  EXPLANATION                │   │
│  [Example 1] [Example 2]      │  │                              │   │
│  [Example 3]                   │  │  "The student correctly     │   │
│                                │  │   identified X and Y but    │   │
│                                │  │   missed concept Z..."      │   │
│                                │  │                              │   │
│                                │  │  ✅ Matched: [concept A]    │   │
│                                │  │             [concept B]     │   │
│                                │  │  ❌ Missing: [concept C]    │   │
│                                │  │  ⚠️  Partial: [concept D]   │   │
│                                │  └────────────────────────────┘   │
│                                │                                    │
│                                │  ┌────────────────────────────┐   │
│                                │  │  PHRASE ALIGNMENT           │   │
│                                │  │                              │   │
│                                │  │  Reference:                  │   │
│                                │  │  "Photosynthesis [converts]  │   │
│                                │  │   [light energy] into        │   │
│                                │  │   [chemical energy]"         │   │
│                                │  │                              │   │
│                                │  │  Student:                    │   │
│                                │  │  "Plants [use sunlight] to   │   │
│                                │  │   [make food]"               │   │
│                                │  │                              │   │
│                                │  │  Legend: 🟢match 🟡partial   │   │
│                                │  │          🔴missing           │   │
│                                │  └────────────────────────────┘   │
├────────────────────────────────┴────────────────────────────────────┤
│  Graded 3 answers this session  •  Avg score: 7.2  •  Model: v2.1 │
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 2: Grading History (Modal / Slide-over)

```
┌──────────────────────────────────────┐
│  📋 Grading History                  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ Q: "What is photosynthesis?"   │  │
│  │ Score: 8.5/10  •  2 min ago    │  │
│  │ [View Details →]               │  │
│  ├────────────────────────────────┤  │
│  │ Q: "Explain Newton's 3rd law"  │  │
│  │ Score: 6.0/10  •  5 min ago    │  │
│  │ [View Details →]               │  │
│  └────────────────────────────────┘  │
│                                      │
│  [Export CSV]  [Clear History]       │
└──────────────────────────────────────┘
```

---

## Key Components

| Component | Type | Details |
|---|---|---|
| Question Input | `<textarea>` | Auto-resize, placeholder with example |
| Reference Answer Input | `<textarea>` | Collapsible, pre-fillable from examples |
| Student Answer Input | `<textarea>` | Primary input, larger, with highlight overlay |
| Grade Button | `<button>` | Amber/gold, loading spinner during inference |
| Score Display | Custom card | Large number + progress bar + confidence badge |
| Classification Badge | Pill/tag | Green (correct), Yellow (partial), Red (incorrect) |
| Similarity Gauge | Radial gauge | Animated, shows semantic similarity 0–1 |
| Similarity Breakdown | Horizontal bars | Semantic / Lexical / Key Concept scores |
| Explanation Panel | Card with text | LLM-generated explanation of the grade |
| Concept Tags | Colored pills | ✅ Matched, ❌ Missing, ⚠️ Partial |
| Phrase Alignment | Annotated text | Side-by-side with color-coded spans |
| Mode Selector | Dropdown/toggle | Classification (3-way) vs Regression (0–10) |
| Example Buttons | Chip buttons | Pre-load demo question/answer pairs |
| History Panel | Slide-over drawer | List of past gradings with scores |

---

## User Flow

```
1. User lands on grading workspace (empty results panel)
        │
2. User types question, reference answer, student answer
   (OR clicks a pre-loaded example)
        │
3. User selects grading mode: Classification or Regression
        │
4. User clicks "✨ Grade Answer"
        │
5. Loading state: button shows spinner, results panel shows skeleton
        │
6. Results reveal (staggered animation):
   a. Score card slides in (number counts up)
   b. Classification badge appears
   c. Similarity gauge animates to value
   d. Explanation text types in
   e. Concept tags fade in
   f. Phrase alignment highlights appear
        │
7. User can:
   - Hover concept tags → highlights in student answer
   - Click "View Details" on similarity → expands breakdown
   - Modify student answer → re-grade
   - Click History → see past results
   - Click Example → load new demo data
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts (gauge, bars) |
| Text Highlighting | Custom React component with `<mark>` spans |
| Animations | Framer Motion |
| State | React useState + useReducer (no global store needed) |
| API | Next.js API routes → Python FastAPI backend |
| ML Backend | FastAPI serving the grading model (SBERT + DeBERTa) |
| Fonts | Inter (body) + Playfair Display (headings) |

---

## Color Palette

```
Background:    #FAFAF8 (warm white)
Card:          #FFFFFF
Primary:       #D97706 (amber-600, grade button)
Correct:       #059669 (emerald-600)
Partial:       #D97706 (amber-600)
Incorrect:     #DC2626 (red-600)
Text:          #1F2937 (gray-800)
Muted:         #6B7280 (gray-500)
Border:        #E5E7EB (gray-200)
```

---

## Wireframe: Score Card Component (Pseudo-JSX)

```tsx
function ScoreCard({ score, maxScore, confidence, label }) {
  return (
    <div className="rounded-xl border bg-white p-6 shadow-sm">
      {/* Score row */}
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-gray-500 uppercase tracking-wide">
          Score
        </span>
        <span className="text-4xl font-bold text-gray-900">
          <CountUp end={score} decimals={1} /> 
          <span className="text-lg text-gray-400">/ {maxScore}</span>
        </span>
      </div>

      {/* Progress bar */}
      <div className="mt-3 h-3 rounded-full bg-gray-100 overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${colorByLabel(label)}`}
          initial={{ width: 0 }}
          animate={{ width: `${(score / maxScore) * 100}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>

      {/* Footer row */}
      <div className="mt-3 flex items-center justify-between">
        <Badge variant={label}>
          {label === "correct" && "✅ Correct"}
          {label === "partial" && "⚠️ Partially Correct"}
          {label === "incorrect" && "❌ Incorrect"}
        </Badge>
        <span className="text-sm text-gray-500">
          Confidence: {confidence}%
        </span>
      </div>
    </div>
  );
}
```

## Wireframe: Phrase Alignment Component (Pseudo-JSX)

```tsx
function PhraseAlignment({ referenceSpans, studentSpans }) {
  return (
    <div className="rounded-xl border bg-white p-6">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-500 mb-4">
        Phrase Alignment
      </h3>

      <div className="space-y-4">
        <div>
          <p className="text-xs text-gray-400 mb-1">Reference Answer</p>
          <p className="text-gray-800 leading-relaxed">
            {referenceSpans.map((span, i) => (
              <span key={i} className={highlightClass(span.match)}>
                {span.text}
              </span>
            ))}
          </p>
        </div>

        <div>
          <p className="text-xs text-gray-400 mb-1">Student Answer</p>
          <p className="text-gray-800 leading-relaxed">
            {studentSpans.map((span, i) => (
              <span key={i} className={highlightClass(span.match)}>
                {span.text}
              </span>
            ))}
          </p>
        </div>
      </div>

      <div className="mt-4 flex gap-4 text-xs text-gray-500">
        <span>🟢 Matched</span>
        <span>🟡 Partial</span>
        <span>🔴 Missing</span>
      </div>
    </div>
  );
}
```
