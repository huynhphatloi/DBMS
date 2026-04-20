# Project 2 — Misconception & Error Pattern Mining

## UI Concept

**"Research Analytics Lab"** — A dark-themed data exploration tool inspired by Jupyter notebooks meets Grafana. This is a power-user analytics interface for researchers to explore clusters of student misconceptions. The design uses a dark background (#0F172A slate-900) with vibrant data visualization colors (cyan, magenta, lime). It feels like a data science workbench — dense with information but well-organized.

Visual identity: Dark mode, neon-accent scatter plots, collapsible panels, data tables with sorting/filtering, and a left sidebar for navigation. The overall feel is "research tool" not "consumer app."

---

## Main Screens

### Screen 1: Cluster Explorer (Primary View)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◉ MisconceptionMiner    [Dataset ▾]  [Embedding ▾]  [Clustering ▾]    │
│                           SciEntsBank   Strategy B     HDBSCAN          │
├──────┬───────────────────────────────────────────────────────────────────┤
│      │                                                                   │
│  📊  │  ┌─────────────────────────────────────────────────────────────┐  │
│  NAV │  │                                                             │  │
│      │  │              UMAP CLUSTER VISUALIZATION                     │  │
│ ───  │  │                                                             │  │
│      │  │         ●●                    ○○○                           │  │
│ 🔍   │  │       ●●●●●                ○○○○○○                          │  │
│ Over │  │      ●●●●●●●●           ○○○○○○○○                           │  │
│ view │  │       ●●●●●              ○○○○○                             │  │
│      │  │         ●●                  ○○                              │  │
│ 📈   │  │                                        ◆◆◆                  │  │
│ Freq │  │              ▲▲▲▲                    ◆◆◆◆◆◆                 │  │
│      │  │            ▲▲▲▲▲▲▲                  ◆◆◆◆◆                  │  │
│ 📋   │  │             ▲▲▲▲▲                    ◆◆◆                   │  │
│ Table│  │              ▲▲                                             │  │
│      │  │                                                             │  │
│ ⚙️   │  │  Clusters: 7  │  Silhouette: 0.62  │  NMI: 0.71           │  │
│ Cfg  │  └─────────────────────────────────────────────────────────────┘  │
│      │                                                                   │
│      │  ┌──────────────────────────┬──────────────────────────────────┐  │
│      │  │  CLUSTER SUMMARY         │  SELECTED CLUSTER: #3           │  │
│      │  │                          │  "Energy Confusion"              │  │
│      │  │  #1 ● Force Direction    │                                  │  │
│      │  │     42 answers, 0.78 coh │  Keywords:                       │  │
│      │  │  #2 ● Unit Confusion     │  [energy] [force] [work]        │  │
│      │  │     38 answers, 0.65 coh │  [power] [heat]                 │  │
│      │  │  #3 ● Energy Confusion   │                                  │  │
│      │  │     55 answers, 0.72 coh │  Example Answers:                │  │
│      │  │  #4 ● Process Reversal   │  ┌────────────────────────────┐ │  │
│      │  │     29 answers, 0.81 coh │  │ "Energy is the same as    │ │  │
│      │  │  #5 ● Scope Error        │  │  force because they both  │ │  │
│      │  │     33 answers, 0.59 coh │  │  make things move"        │ │  │
│      │  │  ...                     │  │  Score: 2/10  Q: q_042    │ │  │
│      │  │                          │  ├────────────────────────────┤ │  │
│      │  │  [-1] ○ Noise            │  │ "Work and energy are the  │ │  │
│      │  │     12 answers           │  │  same thing just measured │ │  │
│      │  │                          │  │  differently"             │ │  │
│      │  │                          │  │  Score: 3/10  Q: q_015    │ │  │
│      │  │                          │  └────────────────────────────┘ │  │
│      │  └──────────────────────────┴──────────────────────────────────┘  │
├──────┴───────────────────────────────────────────────────────────────────┤
│  7 clusters  •  210 answers  •  HDBSCAN (min_cluster=5)  •  UMAP 2D    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Screen 2: Frequency Analysis

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◉ MisconceptionMiner  >  Frequency Analysis                            │
├──────┬───────────────────────────────────────────────────────────────────┤
│      │                                                                   │
│  NAV │  ┌─────────────────────────────────────────────────────────────┐  │
│      │  │  TOP MISCONCEPTION KEYWORDS                                 │  │
│      │  │                                                             │  │
│      │  │  energy ████████████████████████████████  (142)             │  │
│      │  │  force  ██████████████████████████  (118)                   │  │
│      │  │  same   ████████████████████  (89)                          │  │
│      │  │  heat   ██████████████████  (82)                            │  │
│      │  │  move   ████████████████  (71)                              │  │
│      │  │  work   ██████████████  (63)                                │  │
│      │  │  light  ████████████  (55)                                  │  │
│      │  │  power  ██████████  (48)                                    │  │
│      │  └─────────────────────────────────────────────────────────────┘  │
│      │                                                                   │
│      │  ┌──────────────────────────┬──────────────────────────────────┐  │
│      │  │  MISCONCEPTION BY DOMAIN │  MISCONCEPTION BY QUESTION      │  │
│      │  │                          │                                  │  │
│      │  │  [Stacked bar chart]     │  [Heatmap: question × cluster]  │  │
│      │  │  Physics    ████ 45%     │                                  │  │
│      │  │  Biology    ███  30%     │  q_01  ■■■□□□□                  │  │
│      │  │  Chemistry  ██   18%     │  q_02  □■■■■□□                  │  │
│      │  │  Earth Sci  █    7%      │  q_03  □□□■■■■                  │  │
│      │  │                          │  q_04  ■■□□□□■                  │  │
│      │  └──────────────────────────┴──────────────────────────────────┘  │
└──────┴───────────────────────────────────────────────────────────────────┘
```

### Screen 3: Data Table View

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◉ MisconceptionMiner  >  Answer Table                                   │
├──────┬───────────────────────────────────────────────────────────────────┤
│      │  [Search 🔍]  [Filter: Cluster ▾] [Filter: Domain ▾] [Export]    │
│  NAV │                                                                   │
│      │  ┌────┬──────────┬──────────────────────┬────────┬────────────┐  │
│      │  │ ID │ Question │ Student Answer        │ Cluster│ Score      │  │
│      │  ├────┼──────────┼──────────────────────┼────────┼────────────┤  │
│      │  │ 42 │ What is  │ "Energy is the same  │ #3 ●   │ 2/10       │  │
│      │  │    │ energy?  │  as force..."         │        │            │  │
│      │  ├────┼──────────┼──────────────────────┼────────┼────────────┤  │
│      │  │ 43 │ Explain  │ "Heat makes things   │ #3 ●   │ 3/10       │  │
│      │  │    │ heat...  │  have more energy..." │        │            │  │
│      │  ├────┼──────────┼──────────────────────┼────────┼────────────┤  │
│      │  │ 44 │ What is  │ "Gravity pulls down  │ #1 ●   │ 4/10       │  │
│      │  │    │ gravity? │  because of weight"   │        │            │  │
│      │  └────┴──────────┴──────────────────────┴────────┴────────────┘  │
│      │                                                                   │
│      │  Showing 1–25 of 210  [< 1 2 3 4 5 ... 9 >]                     │
└──────┴───────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | Type | Details |
|---|---|---|
| UMAP Scatter Plot | Interactive canvas | Plotly.js or D3, zoom/pan, click to select cluster, tooltip on hover |
| Cluster List | Scrollable sidebar | Color-coded dots, answer count, cohesion score, click to select |
| Cluster Detail Panel | Expandable card | Keywords as tags, example answers as cards, scrollable |
| Keyword Bar Chart | Horizontal bars | Top-N keywords with frequency counts, clickable |
| Domain Distribution | Stacked bar chart | Misconceptions broken down by subject domain |
| Question Heatmap | Grid/matrix | Questions × clusters, color intensity = frequency |
| Data Table | Sortable/filterable | TanStack Table with pagination, search, column filters |
| Config Panel | Collapsible sidebar | Embedding strategy selector, clustering params, n_components |
| Metric Badges | Inline stats | Silhouette, NMI, ARI, Davies-Bouldin displayed as badges |
| Dataset Selector | Dropdown | Switch between SciEntsBank, Data_Generate, etc. |
| Export Button | Button | Download clusters as CSV/JSON |

---

## User Flow

```
1. User lands on Cluster Explorer (default view)
   - UMAP plot loads with pre-computed clusters
   - Cluster summary list populates on the left
        │
2. User can adjust parameters:
   - Change dataset (SciEntsBank / Data_Generate)
   - Change embedding strategy (A: answer-only, B: Q+A, C: full triplet)
   - Change clustering method (KMeans / HDBSCAN / BERTopic)
   → Plot re-renders with new clusters
        │
3. User clicks a cluster dot on the UMAP plot (or clicks cluster in list)
   → Cluster detail panel opens on the right
   → Shows keywords, example answers, cohesion score
   → Corresponding dots on UMAP are highlighted
        │
4. User navigates to Frequency Analysis (sidebar nav)
   → Sees keyword frequency bars
   → Sees domain distribution and question heatmap
   → Clicks a keyword → filters to answers containing it
        │
5. User navigates to Data Table
   → Browses all answers with cluster assignments
   → Filters by cluster, domain, score range
   → Clicks a row → expands to show full answer + question
        │
6. User exports results (CSV/JSON) for thesis
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS (dark mode) + custom CSS for plots |
| Scatter Plot | Plotly.js (react-plotly.js) — best for interactive UMAP |
| Bar Charts | Recharts or Nivo |
| Heatmap | Nivo HeatMap or custom SVG |
| Data Table | TanStack Table (react-table v8) |
| State | Zustand (lightweight global state for selected cluster, filters) |
| API | Next.js API routes → FastAPI backend |
| ML Backend | FastAPI serving pre-computed embeddings + clustering results |
| Fonts | JetBrains Mono (data/code) + Inter (UI text) |

---

## Color Palette (Dark Theme)

```
Background:     #0F172A (slate-900)
Surface:        #1E293B (slate-800)
Card:           #334155 (slate-700)
Border:         #475569 (slate-600)
Text Primary:   #F1F5F9 (slate-100)
Text Secondary: #94A3B8 (slate-400)
Accent Cyan:    #06B6D4 (cyan-500)    — cluster highlights
Accent Magenta: #D946EF (fuchsia-500) — secondary data
Accent Lime:    #84CC16 (lime-500)    — positive metrics
Accent Amber:   #F59E0B (amber-500)   — warnings
Cluster Colors: ["#06B6D4", "#D946EF", "#84CC16", "#F59E0B",
                 "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6"]
```

---

## Wireframe: UMAP Scatter Plot Component (Pseudo-JSX)

```tsx
function UMAPPlot({ clusters, selectedCluster, onClusterSelect }) {
  const traces = clusters.map((cluster) => ({
    x: cluster.points.map(p => p.x),
    y: cluster.points.map(p => p.y),
    mode: "markers",
    type: "scattergl",
    name: cluster.label,
    marker: {
      size: 6,
      color: cluster.color,
      opacity: selectedCluster === null || selectedCluster === cluster.id
        ? 0.8 : 0.15,
      line: { width: selectedCluster === cluster.id ? 2 : 0, color: "#fff" },
    },
    text: cluster.points.map(p => p.studentAnswer.slice(0, 80)),
    hovertemplate: "<b>%{text}</b><br>Cluster: " + cluster.label + "<extra></extra>",
  }));

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          UMAP Cluster Visualization
        </h2>
        <div className="flex gap-2 text-xs text-slate-400">
          <MetricBadge label="Silhouette" value={0.62} />
          <MetricBadge label="NMI" value={0.71} />
          <MetricBadge label="Clusters" value={clusters.length} />
        </div>
      </div>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "#1E293B",
          font: { color: "#94A3B8", family: "Inter" },
          xaxis: { showgrid: false, zeroline: false, showticklabels: false },
          yaxis: { showgrid: false, zeroline: false, showticklabels: false },
          showlegend: true,
          legend: { orientation: "h", y: -0.1, font: { size: 11 } },
          margin: { t: 10, r: 10, b: 40, l: 10 },
        }}
        config={{ responsive: true, displayModeBar: false }}
        onClick={(event) => {
          const clusterIdx = event.points[0]?.curveNumber;
          onClusterSelect(clusters[clusterIdx]?.id ?? null);
        }}
        className="w-full h-[450px]"
      />
    </div>
  );
}
```
