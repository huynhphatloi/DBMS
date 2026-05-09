# Deployment Guide — 4 ASAG Demo Apps

## Architecture

```
GitHub Repo
    ├── demos/shared/                 → Prisma schema + seed script
    ├── demos/project1-grading/       → Vercel App 1 (Grading Dashboard)
    ├── demos/project2-misconception/ → Vercel App 2 (Misconception Mining)
    ├── demos/project3-rubric/        → Vercel App 3 (Rubric Grading)
    └── demos/project4-feedback/      → Vercel App 4 (Feedback Assistant)

Database: Supabase PostgreSQL (free tier, 500MB)
ORM: Prisma (type-safe queries)
ML Inference: HF Inference API (free) + lexical fallback
Frontend: Vercel (free tier)
```

## Step 1: Create Supabase Project

1. Go to https://supabase.com and sign up (free)
2. Click "New Project" → choose a name and password
3. Go to **Settings → Database → Connection string**
4. Copy both:
   - **URI** (Session mode / port 5432) → this is `DIRECT_URL`
   - **URI** (Transaction mode / port 6543) → this is `DATABASE_URL`

## Step 2: Setup Database

```bash
cd demos/shared
npm install

# Create .env with your Supabase credentials
cp .env.example .env
# Edit .env → paste your DATABASE_URL and DIRECT_URL

# Push schema to Supabase + seed data
npm run db:setup
```

This will:
- Create all tables (student_answers, grading_results, clusters, feedback_results)
- Load 10,000 records from data-generate.csv
- Load 129 records from data-scraping.json

## Step 3: Setup Demo Apps

```bash
cd demos
bash setup.sh
```

Or manually for each app:
```bash
cd demos/project1-grading
npm install
npm install @prisma/client
cp ../shared/prisma/schema.prisma prisma/
cp ../shared/lib/db.ts lib/
cp .env.example .env.local
# Edit .env.local → paste DATABASE_URL, DIRECT_URL, HF_API_TOKEN
npx prisma generate
npm run dev
```

## Deploy to Vercel

### Push to GitHub

```bash
git add .
git commit -m "ASAG demos with Supabase + Prisma"
git push origin main
```

### Create 4 Vercel Projects

Go to https://vercel.com/new → import your GitHub repo.
Create **4 separate projects**, one per demo:

| Project | Root Directory | Framework |
|---|---|---|
| asag-grading | `demos/project1-grading` | Next.js |
| asag-misconception | `demos/project2-misconception` | Next.js |
| asag-rubric | `demos/project3-rubric` | Next.js |
| asag-feedback | `demos/project4-feedback` | Next.js |

For each project, add these **Environment Variables** in Vercel Settings:

| Variable | Value |
|---|---|
| `DATABASE_URL` | Your Supabase pooler URL (port 6543) |
| `DIRECT_URL` | Your Supabase direct URL (port 5432) |
| `HF_API_TOKEN` | (optional) Hugging Face free token |

### Build Command Override

In each Vercel project settings, set:
- **Build Command:** `npx prisma generate && next build`
- This ensures Prisma client is generated before the Next.js build

## Run Locally

```bash
# 1. Setup database first
cd demos/shared && npm install && npm run db:setup && cd ..

# 2. Setup all apps
bash setup.sh

# 3. Run all 4
cd project1-grading && npm run dev &
cd ../project2-misconception && npm run dev &
cd ../project3-rubric && npm run dev &
cd ../project4-feedback && npm run dev &

# Open:
# http://localhost:3001  — Grading Dashboard
# http://localhost:3002  — Misconception Mining
# http://localhost:3003  — Rubric Grading
# http://localhost:3004  — Feedback Assistant
```

| Feature | Status |
|---|---|
| Lexical metrics (Jaccard, Word Overlap, ROUGE-L) | ✅ Real computation |
| SBERT semantic similarity | ✅ Real (with HF token) / Approximated (without) |
| Phrase alignment highlighting | ✅ Real token matching |
| Concept gap detection | ⚡ Keyword-based (real NLI needs HF token) |
| UMAP cluster visualization | 📊 Pre-computed mock data (real needs Python backend) |
| T5 feedback generation | 📝 Template-based (real T5 needs GPU) |
| Score bars and charts | ✅ Real Recharts rendering |

## Cost

| Service | Cost | What you get |
|---|---|---|
| Vercel Free | $0 | 4 Next.js apps, custom domains, HTTPS |
| Hugging Face Free | $0 | ~30K inference calls/month |
| GitHub Free | $0 | Unlimited public repos |
| **Total** | **$0** | |


## Database Schema (Prisma ORM)

```
Tables:
├── student_answers      — 10,129 records (UnifiedRecord schema)
│   ├── 37 columns matching Python dataclass
│   ├── Indexes: (source_dataset, split), question_id, label_3way, domain
│   └── Relations → grading_results, cluster_assignments
├── grading_results      — Stores grading predictions per model
├── clusters             — Cluster metadata (labels, keywords, metrics)
├── cluster_assignments  — Maps student_answers → clusters with UMAP coords
└── feedback_results     — Stores generated feedback with quality scores
```

## Cost (Updated)

| Service | Cost | What you get |
|---|---|---|
| Supabase Free | $0 | PostgreSQL 500MB, REST API, Auth |
| Vercel Free | $0 | 4 Next.js apps, custom domains, HTTPS |
| Hugging Face Free | $0 | ~30K inference calls/month |
| GitHub Free | $0 | Unlimited public repos |
| **Total** | **$0** | |
