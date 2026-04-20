# ASAG Thesis UI Demos — Overview

## 4 Projects, 4 Distinct Identities

| # | Project | UI Concept | Visual Style | Primary Layout |
|---|---------|-----------|-------------|----------------|
| 1 | Auto Grading | Teacher's Grading Desk | Warm light, amber accents, serif headings | Split panel (input ↔ results) |
| 2 | Misconception Mining | Research Analytics Lab | Dark mode, neon data viz, dense panels | Sidebar nav + canvas + detail panel |
| 3 | Rubric Grading | Academic Evaluation Console | Clean white, blue primary, structured grids | Step wizard (setup → grade → results) |
| 4 | Feedback Generation | Study Buddy Assistant | Gradient bg, violet accents, rounded cards | Chat-style (bubbles + feedback card) |

---

## Visual Differentiation Matrix

| Aspect | Project 1 | Project 2 | Project 3 | Project 4 |
|--------|-----------|-----------|-----------|-----------|
| Theme | Light (warm) | Dark | Light (cool) | Light (gradient) |
| Background | #FAFAF8 cream | #0F172A slate | #F8FAFC white | violet→blue gradient |
| Primary Color | #D97706 amber | #06B6D4 cyan | #2563EB blue | #7C3AED violet |
| Font Vibe | Academic serif | Monospace data | Professional sans | Friendly rounded |
| Heading Font | Playfair Display | JetBrains Mono | Source Serif 4 | Nunito |
| Body Font | Inter | Inter | Inter | Nunito |
| Layout | 50/50 split | Sidebar + canvas | Wizard steps | Vertical scroll |
| Interaction | Form → results | Explore → drill down | Configure → evaluate | Submit → read |
| Persona | Teacher | Researcher | Administrator | Student |
| Density | Medium | High | Medium | Low |

---

## Shared Tech Stack

All 4 projects share a common foundation:

```
Frontend:   Next.js 14 (App Router) + React 18
Styling:    Tailwind CSS v3 + shadcn/ui components
Charts:     Recharts (Projects 1, 3, 4) + Plotly.js (Project 2)
Animation:  Framer Motion
API Layer:  Next.js API Routes → FastAPI (Python) backend
ML Models:  Served via FastAPI (SBERT, DeBERTa, T5, NLI models)
```

---

## Recommended Demo Order for Thesis Presentation

1. **Project 1 (Grading)** — Start with the core problem. Show how the system grades a student answer. This is the foundation.

2. **Project 2 (Misconception Mining)** — "But what patterns do we see across many wrong answers?" Transition from individual grading to aggregate analysis.

3. **Project 3 (Rubric Grading)** — "How do we make grading transparent and criteria-based?" Show the explainability angle.

4. **Project 4 (Feedback)** — "Now that we can grade and explain, how do we help the student improve?" End with the student-facing output.

This order tells a story: **Grade → Analyze → Explain → Help**.

---

## File Structure for Each Demo

```
project-N-name/
├── app/
│   ├── layout.tsx          # Root layout with fonts, metadata
│   ├── page.tsx            # Main page
│   ├── api/
│   │   └── grade/route.ts  # API endpoint (proxies to FastAPI)
│   └── globals.css         # Tailwind + custom styles
├── components/
│   ├── ui/                 # shadcn/ui components
│   └── [feature]/          # Feature-specific components
├── lib/
│   └── api.ts              # API client functions
├── public/
│   └── ...                 # Static assets
├── tailwind.config.ts
├── next.config.js
└── package.json
```

---

## Quick Start (for each project)

```bash
npx create-next-app@latest project-name --typescript --tailwind --app
cd project-name
npx shadcn@latest init
npx shadcn@latest add button card input textarea badge slider
npm install recharts framer-motion
# Project 2 only:
npm install react-plotly.js plotly.js @tanstack/react-table zustand
# Project 4 only:
npm install react-type-animation
```
