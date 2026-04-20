# Project 3 — Rubric-based Explainable Grading

## UI Concept

**"Academic Evaluation Console"** — A structured, form-driven evaluation tool that feels like an official educational assessment system. Think Turnitin meets a university LMS grading interface. The design uses a clean white background with a strong blue primary color (#2563EB), structured grid layouts, and a step-by-step wizard feel. The rubric table is the centerpiece — editable, weighted, and visually connected to the scoring output.

Visual identity: Crisp borders, structured grids, blue accent color, step indicators, professional typography. The UI communicates "institutional quality" and "transparency in grading."

---

## Main Screens

### Screen 1: Rubric Setup + Grading (Primary — Multi-Panel Layout)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  🎓 RubricGrader          Step: [1 Setup ●──2 Grade ○──3 Results ○]    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─── STEP 1: SETUP ────────────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  Question                                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐ │   │
│  │  │ Explain the process of photosynthesis and its importance    │ │   │
│  │  │ for life on Earth.                                          │ │   │
│  │  └──────────────────────────────────────────────────────────────┘ │   │
│  │                                                                    │   │
│  │  Rubric                                              [+ Add Row]  │   │
│  │  ┌──────────────────────┬────────┬────────────────────────────┐  │   │
│  │  │ Criterion            │ Weight │ Description                │  │   │
│  │  ├──────────────────────┼────────┼────────────────────────────┤  │   │
│  │  │ Core Process         │ 30%  ▾ │ Mentions light energy →    │  │   │
│  │  │ [editable]           │[slider]│ chemical energy conversion │  │   │
│  │  ├──────────────────────┼────────┼────────────────────────────┤  │   │
│  │  │ Key Components       │ 25%  ▾ │ Identifies chlorophyll,    │  │   │
│  │  │ [editable]           │[slider]│ CO2, H2O, glucose, O2     │  │   │
│  │  ├──────────────────────┼────────┼────────────────────────────┤  │   │
│  │  │ Importance           │ 25%  ▾ │ Explains role in food      │  │   │
│  │  │ [editable]           │[slider]│ chains and oxygen supply   │  │   │
│  │  ├──────────────────────┼────────┼────────────────────────────┤  │   │
│  │  │ Scientific Language  │ 20%  ▾ │ Uses correct terminology   │  │   │
│  │  │ [editable]           │[slider]│ and clear explanations     │  │   │
│  │  └──────────────────────┴────────┴────────────────────────────┘  │   │
│  │                                                                    │   │
│  │  Total Weight: [████████████████████████████████████████] 100%    │   │
│  │                                                                    │   │
│  │  Student Answer                                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐ │   │
│  │  │ Plants use sunlight to make food. They take in carbon       │ │   │
│  │  │ dioxide and water and produce glucose. This is important    │ │   │
│  │  │ because animals eat plants.                                 │ │   │
│  │  └──────────────────────────────────────────────────────────────┘ │   │
│  │                                                                    │   │
│  │  [Load Example ▾]              [ ▶ EVALUATE WITH RUBRIC ]        │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Screen 2: Results View (After Evaluation)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  🎓 RubricGrader          Step: [1 Setup ●──2 Grade ●──3 Results ●]    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────┬─────────────────────────────────────┐ │
│  │  SCORE BREAKDOWN             │  VISUAL ANALYSIS                    │ │
│  │                              │                                     │ │
│  │  ┌────────────────────────┐  │  ┌─────────────────────────────┐   │ │
│  │  │  FINAL SCORE           │  │  │                             │   │ │
│  │  │                        │  │  │     RADAR CHART             │   │ │
│  │  │     6.5 / 10           │  │  │                             │   │ │
│  │  │  ████████████░░░░░░░░  │  │  │    Core Process             │   │ │
│  │  │                        │  │  │         ╱╲                   │   │ │
│  │  │  Grade: B-             │  │  │        ╱  ╲                  │   │ │
│  │  └────────────────────────┘  │  │  Lang ╱    ╲ Components     │   │ │
│  │                              │  │       ╲    ╱                 │   │ │
│  │  Per-Criterion Scores:       │  │        ╲  ╱                  │   │ │
│  │                              │  │         ╲╱                   │   │ │
│  │  Core Process     (30%)      │  │      Importance              │   │ │
│  │  ████████████████░░  8/10    │  │                             │   │ │
│  │                              │  │  ── scored ── max ──        │   │ │
│  │  Key Components   (25%)      │  └─────────────────────────────┘   │ │
│  │  ██████████░░░░░░░░  5/10    │                                     │ │
│  │                              │  ┌─────────────────────────────┐   │ │
│  │  Importance       (25%)      │  │  STACKED BAR BREAKDOWN      │   │ │
│  │  ████████████████░░  8/10    │  │                             │   │ │
│  │                              │  │  ┌──┬──┬──┬──┐             │   │ │
│  │  Scientific Lang  (20%)      │  │  │CP│KC│IM│SL│  = 6.5     │   │ │
│  │  ██████░░░░░░░░░░░░  3/10    │  │  │2.4│1.3│2.0│0.6│         │   │ │
│  │                              │  │  └──┴──┴──┴──┘             │   │ │
│  │  Weighted:                   │  │  (weighted contributions)   │   │ │
│  │  2.4 + 1.25 + 2.0 + 0.6     │  └─────────────────────────────┘   │ │
│  │  = 6.25 → 6.5               │                                     │ │
│  └──────────────────────────────┴─────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  EXPLAINABILITY: Answer ↔ Rubric Alignment                        │ │
│  │                                                                    │ │
│  │  Student Answer (annotated):                                       │ │
│  │                                                                    │ │
│  │  "[Plants use sunlight to make food]₁. [They take in carbon       │ │
│  │   dioxide and water]₂ and [produce glucose]₂. [This is important  │ │
│  │   because animals eat plants]₃."                                   │ │
│  │                                                                    │ │
│  │  ₁ → Core Process (partial match: mentions sunlight→food but      │ │
│  │       not "light energy → chemical energy" conversion)             │ │
│  │  ₂ → Key Components (mentions CO2, H2O, glucose but misses        │ │
│  │       chlorophyll and O2)                                          │ │
│  │  ₃ → Importance (mentions food chain but not oxygen supply)        │ │
│  │  ⚠  Scientific Language: informal language ("make food" instead    │ │
│  │     of "synthesize glucose")                                       │ │
│  │                                                                    │ │
│  │  [No match found for: chlorophyll, oxygen production, light        │ │
│  │   energy, chemical energy]                                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  [← Back to Setup]  [Export PDF]  [Grade Another]                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | Type | Details |
|---|---|---|
| Step Indicator | Progress bar | 3 steps: Setup → Grade → Results, clickable |
| Question Input | `<textarea>` | Standard text input |
| Rubric Table | Editable table | Rows: criterion name, weight (slider 0–100%), description |
| Weight Slider | Range input | Per-criterion, with total weight validation bar |
| Weight Total Bar | Progress bar | Shows sum of weights, turns red if ≠ 100% |
| Add/Remove Row | Buttons | + Add criterion, × remove (min 2 rows) |
| Student Answer Input | `<textarea>` | Primary input area |
| Evaluate Button | `<button>` | Blue primary, triggers evaluation |
| Final Score Card | Large display | Score / max, letter grade, progress bar |
| Per-Criterion Bars | Horizontal bars | Score per criterion with weight label |
| Radar Chart | Recharts RadarChart | Multi-axis showing score per criterion |
| Stacked Bar | Recharts StackedBar | Weighted contribution of each criterion |
| Explainability Panel | Annotated text | Student answer with subscript markers linking to rubric criteria |
| Alignment Legend | Color-coded list | Which text spans match which criteria |
| Example Loader | Dropdown | Pre-built rubric + question + answer combos |
| Export PDF | Button | Generate a printable evaluation report |

---

## User Flow

```
1. User lands on Step 1: Setup
        │
2. User enters (or loads example):
   a. Question text
   b. Rubric criteria (name, weight, description)
      - Can add/remove rows
      - Weights must sum to 100% (validated in real-time)
   c. Student answer
        │
3. User clicks "▶ Evaluate with Rubric"
        │
4. Loading: step indicator advances, skeleton results appear
        │
5. Step 3: Results reveal:
   a. Final score card (animated count-up)
   b. Per-criterion score bars (staggered animation)
   c. Radar chart renders
   d. Stacked bar shows weighted contributions
   e. Explainability panel shows annotated answer
        │
6. User explores results:
   - Hover criterion bar → highlights matching text in answer
   - Click radar chart axis → scrolls to that criterion's explanation
   - Read alignment annotations to understand scoring
        │
7. User can:
   - Go back to modify rubric/answer and re-evaluate
   - Export results as PDF
   - Grade another answer with same rubric
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS + shadcn/ui |
| Radar Chart | Recharts (RadarChart component) |
| Stacked Bar | Recharts (BarChart, stacked) |
| Editable Table | Custom React component with contentEditable or inline inputs |
| Weight Sliders | shadcn/ui Slider component |
| Text Annotation | Custom React component with indexed `<span>` markers |
| Animations | Framer Motion (step transitions, score reveals) |
| PDF Export | html2canvas + jsPDF (or react-pdf) |
| State | React useState (form state) + useReducer (rubric rows) |
| API | Next.js API routes → FastAPI backend |
| ML Backend | FastAPI with NLI model for criterion matching |
| Fonts | Inter (body) + Source Serif 4 (headings) |

---

## Color Palette

```
Background:     #F8FAFC (slate-50)
Card:           #FFFFFF
Primary:        #2563EB (blue-600)
Primary Light:  #DBEAFE (blue-100)
Success:        #059669 (emerald-600)
Warning:        #D97706 (amber-600)
Danger:         #DC2626 (red-600)
Text:           #0F172A (slate-900)
Muted:          #64748B (slate-500)
Border:         #E2E8F0 (slate-200)
Criterion Colors: ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626"]
```

---

## Wireframe: Rubric Table Component (Pseudo-JSX)

```tsx
function RubricTable({ criteria, onUpdate, onAdd, onRemove }) {
  const totalWeight = criteria.reduce((sum, c) => sum + c.weight, 0);
  const isValid = Math.abs(totalWeight - 100) < 0.01;

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <h3 className="text-sm font-semibold text-slate-700">Rubric Criteria</h3>
        <button onClick={onAdd} className="text-sm text-blue-600 hover:text-blue-800">
          + Add Criterion
        </button>
      </div>

      <table className="w-full">
        <thead>
          <tr className="border-b bg-slate-50 text-xs text-slate-500 uppercase">
            <th className="px-4 py-2 text-left">Criterion</th>
            <th className="px-4 py-2 text-center w-32">Weight</th>
            <th className="px-4 py-2 text-left">Description</th>
            <th className="px-4 py-2 w-10"></th>
          </tr>
        </thead>
        <tbody>
          {criteria.map((criterion, idx) => (
            <tr key={idx} className="border-b last:border-0">
              <td className="px-4 py-3">
                <input
                  value={criterion.name}
                  onChange={(e) => onUpdate(idx, "name", e.target.value)}
                  className="w-full border-0 bg-transparent font-medium text-slate-800
                             focus:outline-none focus:ring-2 focus:ring-blue-200 rounded px-1"
                />
              </td>
              <td className="px-4 py-3 text-center">
                <div className="flex items-center gap-2">
                  <Slider
                    value={[criterion.weight]}
                    onValueChange={([v]) => onUpdate(idx, "weight", v)}
                    max={100} step={5}
                    className="w-20"
                  />
                  <span className="text-sm font-mono text-slate-600 w-10">
                    {criterion.weight}%
                  </span>
                </div>
              </td>
              <td className="px-4 py-3">
                <input
                  value={criterion.description}
                  onChange={(e) => onUpdate(idx, "description", e.target.value)}
                  className="w-full border-0 bg-transparent text-sm text-slate-600
                             focus:outline-none focus:ring-2 focus:ring-blue-200 rounded px-1"
                />
              </td>
              <td className="px-4 py-3">
                {criteria.length > 2 && (
                  <button onClick={() => onRemove(idx)}
                    className="text-slate-400 hover:text-red-500">×</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Weight total bar */}
      <div className="px-4 py-3 border-t bg-slate-50">
        <div className="flex items-center justify-between text-sm mb-1">
          <span className="text-slate-500">Total Weight</span>
          <span className={isValid ? "text-emerald-600 font-medium" : "text-red-600 font-medium"}>
            {totalWeight}% {isValid ? "✓" : "(must equal 100%)"}
          </span>
        </div>
        <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              isValid ? "bg-emerald-500" : totalWeight > 100 ? "bg-red-500" : "bg-amber-500"
            }`}
            style={{ width: `${Math.min(totalWeight, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
```

## Wireframe: Explainability Panel (Pseudo-JSX)

```tsx
function ExplainabilityPanel({ answer, alignments, criteria }) {
  // alignments: [{ start, end, criterionIdx, matchType }]
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-6">
      <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-4">
        Explainability: Answer ↔ Rubric Alignment
      </h3>

      <div className="mb-4">
        <p className="text-xs text-slate-400 mb-2">Student Answer (annotated):</p>
        <p className="text-slate-800 leading-relaxed text-[15px]">
          <AnnotatedText text={answer} alignments={alignments} criteria={criteria} />
        </p>
      </div>

      <div className="space-y-3 mt-6">
        {alignments.map((a, i) => (
          <div key={i} className="flex gap-3 text-sm">
            <span
              className="inline-block w-6 h-6 rounded-full text-center text-xs
                         leading-6 text-white font-bold shrink-0"
              style={{ backgroundColor: criteria[a.criterionIdx].color }}
            >
              {a.criterionIdx + 1}
            </span>
            <div>
              <span className="font-medium text-slate-700">
                → {criteria[a.criterionIdx].name}
              </span>
              <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${
                a.matchType === "full" ? "bg-emerald-100 text-emerald-700" :
                a.matchType === "partial" ? "bg-amber-100 text-amber-700" :
                "bg-red-100 text-red-700"
              }`}>
                {a.matchType}
              </span>
              <p className="text-slate-500 mt-0.5">{a.explanation}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```
