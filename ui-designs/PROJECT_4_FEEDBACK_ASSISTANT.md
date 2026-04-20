# Project 4 — Feedback Generation System

## UI Concept

**"Study Buddy — AI Learning Assistant"** — A friendly, conversational interface that feels like a personal tutor. Think ChatGPT meets Duolingo. The design uses a chat-like layout with a warm gradient background (soft purple → blue), rounded cards, emoji-style icons, and a supportive tone throughout. This is the most "consumer-facing" of the four demos — it should feel approachable and encouraging, not clinical.

Visual identity: Rounded corners everywhere, gradient accents, card-based feedback sections with icons (💪 Strengths, 🔍 Weaknesses, 💡 Suggestions), a floating input area at the bottom, and smooth slide-in animations. The overall feel is "friendly AI tutor."

---

## Main Screens

### Screen 1: Feedback Interface (Primary — Chat-Style Layout)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  🎓 StudyBuddy                                                     │  │
│  │  Your AI Learning Assistant                                        │  │
│  │                                                                    │  │
│  │  Tone: [😊 Friendly ●] [📚 Academic ○] [📏 Strict ○]             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  ┌─ QUESTION ──────────────────────────────────────────────────┐  │  │
│  │  │  📝 Explain how photosynthesis converts light energy into   │  │  │
│  │  │     chemical energy stored in glucose.                      │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │                          ┌─ YOUR ANSWER ───────────────────────┐  │  │
│  │                          │  Plants use sunlight to make food.  │  │  │
│  │                          │  They take in CO2 and water and     │  │  │
│  │                          │  produce glucose. Animals eat       │  │  │
│  │                          │  plants for energy.                 │  │  │
│  │                          └─────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌─ STUDYBUDDY FEEDBACK ───────────────────────────────────────┐  │  │
│  │  │                                                              │  │  │
│  │  │  Hey! Good effort on this one. You've got the basics        │  │  │
│  │  │  down — let me break it down for you:                       │  │  │
│  │  │                                                              │  │  │
│  │  │  ┌─────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  💪 STRENGTHS                                        │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  ✅ You correctly identified that plants use          │    │  │  │
│  │  │  │     sunlight as an energy source                     │    │  │  │
│  │  │  │  ✅ You mentioned the key inputs: CO2 and water      │    │  │  │
│  │  │  │  ✅ You identified glucose as the output product     │    │  │  │
│  │  │  │  ✅ You connected photosynthesis to the food chain   │    │  │  │
│  │  │  └─────────────────────────────────────────────────────┘    │  │  │
│  │  │                                                              │  │  │
│  │  │  ┌─────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  🔍 AREAS TO IMPROVE                                │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  ⚠️ You didn't explain the energy conversion         │    │  │  │
│  │  │  │     process (light energy → chemical energy)         │    │  │  │
│  │  │  │  ⚠️ Missing: role of chlorophyll in capturing        │    │  │  │
│  │  │  │     light energy                                     │    │  │  │
│  │  │  │  ⚠️ Missing: oxygen (O2) as a byproduct             │    │  │  │
│  │  │  │  ⚠️ "Make food" is too informal — use "synthesize    │    │  │  │
│  │  │  │     glucose" for scientific accuracy                 │    │  │  │
│  │  │  └─────────────────────────────────────────────────────┘    │  │  │
│  │  │                                                              │  │  │
│  │  │  ┌─────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  💡 SUGGESTIONS                                      │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  1. Add the chemical equation:                       │    │  │  │
│  │  │  │     6CO2 + 6H2O + light → C6H12O6 + 6O2            │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  2. Explain that chlorophyll in chloroplasts         │    │  │  │
│  │  │  │     absorbs light energy                             │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  3. Mention that the chemical energy is stored       │    │  │  │
│  │  │  │     in the bonds of glucose molecules                │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  4. Include oxygen as a crucial byproduct            │    │  │  │
│  │  │  └─────────────────────────────────────────────────────┘    │  │  │
│  │  │                                                              │  │  │
│  │  │  ┌─────────────────────────────────────────────────────┐    │  │  │
│  │  │  │  📊 YOUR SCORE                                       │    │  │  │
│  │  │  │                                                      │    │  │  │
│  │  │  │  Completeness   ████████░░░░  65%                    │    │  │  │
│  │  │  │  Accuracy       ██████████░░  80%                    │    │  │  │
│  │  │  │  Terminology    ████░░░░░░░░  35%                    │    │  │  │
│  │  │  │  Overall        ████████░░░░  60%                    │    │  │  │
│  │  │  └─────────────────────────────────────────────────────┘    │  │  │
│  │  │                                                              │  │  │
│  │  │  Was this feedback helpful?  [👍 Yes]  [👎 No]              │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  📝 Question                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │ [Enter the question here...]                                 │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  │                                                                    │  │
│  │  ✏️ Your Answer                                                    │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │ [Type your answer here...]                                   │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  │                                                                    │  │
│  │  [Load Example ▾]                    [ 🚀 Get Feedback ]         │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Screen 2: Tone Comparison View (Side-by-Side)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  🎓 StudyBuddy  >  Tone Comparison                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Same answer, three tones:                                               │
│                                                                          │
│  ┌────────────────────┬────────────────────┬────────────────────┐       │
│  │  😊 FRIENDLY        │  📚 ACADEMIC       │  📏 STRICT         │       │
│  │                     │                    │                    │       │
│  │  "Hey! Nice try!    │  "The response     │  "The answer is   │       │
│  │   You got the       │   demonstrates     │   incomplete.     │       │
│  │   basics right.     │   partial under-   │   Missing: energy │       │
│  │   Let's work on     │   standing of the  │   conversion,     │       │
│  │   adding more       │   photosynthetic   │   chlorophyll,    │       │
│  │   detail..."        │   process..."      │   O2 output.      │       │
│  │                     │                    │   Score: 6/10."   │       │
│  │  Strengths: ...     │  Strengths: ...    │  Strengths: ...   │       │
│  │  Weaknesses: ...    │  Weaknesses: ...   │  Weaknesses: ...  │       │
│  │  Suggestions: ...   │  Suggestions: ...  │  Suggestions: ... │       │
│  └────────────────────┴────────────────────┴────────────────────┘       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Screen 3: History / Progress View

```
┌──────────────────────────────────────────────────────────────────────────┐
│  🎓 StudyBuddy  >  Your Progress                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  📈 Score Trend (last 5 submissions)                               │ │
│  │                                                                    │ │
│  │  10 ┤                                                              │ │
│  │   8 ┤          ●───●                                               │ │
│  │   6 ┤     ●───●         ●                                         │ │
│  │   4 ┤                                                              │ │
│  │   2 ┤ ●                                                           │ │
│  │   0 ┼────┼────┼────┼────┼────                                     │ │
│  │      #1   #2   #3   #4   #5                                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Recent Submissions                                                │ │
│  │                                                                    │ │
│  │  #5  "Explain photosynthesis..."     6/10  •  2 min ago  [View]   │ │
│  │  #4  "What is Newton's 3rd law?"     8/10  •  1 hr ago   [View]   │ │
│  │  #3  "Describe cell division..."     7/10  •  2 hrs ago  [View]   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | Type | Details |
|---|---|---|
| Header | Fixed top bar | Logo, title, tone selector as pill toggle |
| Tone Selector | Pill toggle group | 3 options: Friendly / Academic / Strict, with emoji icons |
| Question Display | Chat bubble (left) | Gray background, left-aligned, shows the question |
| Student Answer | Chat bubble (right) | Blue/purple background, right-aligned |
| Feedback Card | Large card | Contains all 4 sections below, slide-in animation |
| Strengths Section | Card section | Green accent, ✅ checkmarks, list of positive points |
| Weaknesses Section | Card section | Amber accent, ⚠️ warnings, list of gaps |
| Suggestions Section | Card section | Blue accent, 💡 numbered list of improvements |
| Score Bars | Horizontal progress | Completeness, Accuracy, Terminology, Overall |
| Feedback Helpfulness | Thumbs up/down | Quick feedback on the AI's feedback quality |
| Input Area | Fixed bottom panel | Question textarea + answer textarea + submit button |
| Example Loader | Dropdown | Pre-built question + answer pairs |
| Tone Comparison | 3-column layout | Same answer graded in all 3 tones side-by-side |
| Progress Chart | Line chart | Score trend over submissions |
| History List | Card list | Past submissions with scores and timestamps |

---

## User Flow

```
1. User lands on the feedback interface
   - Sees welcoming header with StudyBuddy branding
   - Input area at the bottom is ready
        │
2. User selects tone: Friendly / Academic / Strict
   (default: Friendly)
        │
3. User enters question and student answer
   (OR clicks "Load Example" for a demo)
        │
4. User clicks "🚀 Get Feedback"
        │
5. Loading: rocket animation, "StudyBuddy is thinking..." message
        │
6. Feedback appears (staggered animation):
   a. Question appears as left chat bubble
   b. Student answer appears as right chat bubble
   c. Feedback card slides up:
      - Opening message types in (typewriter effect)
      - Strengths section fades in (green)
      - Weaknesses section fades in (amber)
      - Suggestions section fades in (blue)
      - Score bars animate to their values
   d. Helpfulness prompt appears
        │
7. User can:
   - Change tone → feedback regenerates in new tone
   - Click "Tone Comparison" → see all 3 tones side-by-side
   - Click 👍/👎 on feedback quality
   - Submit another answer
   - View progress/history
        │
8. Optional: User navigates to Progress view
   - Sees score trend chart
   - Reviews past submissions
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS + custom gradient utilities |
| UI Components | shadcn/ui (buttons, cards, toggles) |
| Charts | Recharts (line chart for progress, bar for scores) |
| Animations | Framer Motion (slide-in cards, typewriter text) |
| Typewriter Effect | Custom hook or `react-type-animation` |
| State | React useState (simple form + feedback state) |
| API | Next.js API routes → FastAPI backend |
| ML Backend | FastAPI serving T5 feedback model + concept gap detector |
| Fonts | Nunito (body — friendly, rounded) + Inter (data) |

---

## Color Palette

```
Background:     Linear gradient: #F5F3FF → #EFF6FF (violet-50 → blue-50)
Card:           #FFFFFF (with subtle shadow)
Primary:        #7C3AED (violet-600)
Primary Light:  #EDE9FE (violet-100)
Strengths:      #059669 (emerald-600) / #ECFDF5 (emerald-50)
Weaknesses:     #D97706 (amber-600) / #FFFBEB (amber-50)
Suggestions:    #2563EB (blue-600) / #EFF6FF (blue-50)
Score Bar BG:   #F1F5F9 (slate-100)
Text:           #1E1B4B (indigo-950)
Muted:          #6B7280 (gray-500)
User Bubble:    #7C3AED → #6D28D9 (violet gradient)
Bot Bubble:     #F8FAFC (slate-50)
```

---

## Wireframe: Feedback Card Component (Pseudo-JSX)

```tsx
function FeedbackCard({ feedback, tone }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="rounded-2xl bg-white shadow-lg border border-slate-100 p-6 space-y-5"
    >
      {/* Opening message */}
      <TypewriterText
        text={feedback.openingMessage}
        className="text-slate-700 text-[15px] leading-relaxed"
        speed={30}
      />

      {/* Strengths */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="rounded-xl bg-emerald-50 border border-emerald-200 p-4"
      >
        <h3 className="text-sm font-bold text-emerald-700 mb-3 flex items-center gap-2">
          💪 Strengths
        </h3>
        <ul className="space-y-2">
          {feedback.strengths.map((s, i) => (
            <li key={i} className="flex gap-2 text-sm text-emerald-800">
              <span className="text-emerald-500 shrink-0">✅</span>
              {s}
            </li>
          ))}
        </ul>
      </motion.div>

      {/* Weaknesses */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5 }}
        className="rounded-xl bg-amber-50 border border-amber-200 p-4"
      >
        <h3 className="text-sm font-bold text-amber-700 mb-3 flex items-center gap-2">
          🔍 Areas to Improve
        </h3>
        <ul className="space-y-2">
          {feedback.weaknesses.map((w, i) => (
            <li key={i} className="flex gap-2 text-sm text-amber-800">
              <span className="text-amber-500 shrink-0">⚠️</span>
              {w}
            </li>
          ))}
        </ul>
      </motion.div>

      {/* Suggestions */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.7 }}
        className="rounded-xl bg-blue-50 border border-blue-200 p-4"
      >
        <h3 className="text-sm font-bold text-blue-700 mb-3 flex items-center gap-2">
          💡 Suggestions
        </h3>
        <ol className="space-y-2 list-decimal list-inside">
          {feedback.suggestions.map((s, i) => (
            <li key={i} className="text-sm text-blue-800">{s}</li>
          ))}
        </ol>
      </motion.div>

      {/* Score bars */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.9 }}
        className="rounded-xl bg-slate-50 border border-slate-200 p-4"
      >
        <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
          📊 Your Score
        </h3>
        <div className="space-y-3">
          {feedback.scores.map((score) => (
            <div key={score.label}>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>{score.label}</span>
                <span>{score.value}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-violet-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${score.value}%` }}
                  transition={{ duration: 0.8, delay: 1.0 }}
                />
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Helpfulness */}
      <div className="flex items-center justify-center gap-4 pt-2 text-sm text-slate-500">
        <span>Was this feedback helpful?</span>
        <button className="px-3 py-1 rounded-full border hover:bg-emerald-50
                           hover:border-emerald-300 transition">
          👍 Yes
        </button>
        <button className="px-3 py-1 rounded-full border hover:bg-red-50
                           hover:border-red-300 transition">
          👎 No
        </button>
      </div>
    </motion.div>
  );
}
```

## Wireframe: Tone Selector (Pseudo-JSX)

```tsx
function ToneSelector({ selected, onChange }) {
  const tones = [
    { id: "friendly",  emoji: "😊", label: "Friendly" },
    { id: "academic",  emoji: "📚", label: "Academic" },
    { id: "strict",    emoji: "📏", label: "Strict" },
  ];

  return (
    <div className="flex items-center gap-1 bg-slate-100 rounded-full p-1">
      {tones.map((tone) => (
        <button
          key={tone.id}
          onClick={() => onChange(tone.id)}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
            selected === tone.id
              ? "bg-white text-violet-700 shadow-sm"
              : "text-slate-500 hover:text-slate-700"
          }`}
        >
          {tone.emoji} {tone.label}
        </button>
      ))}
    </div>
  );
}
```
