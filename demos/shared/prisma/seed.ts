/**
 * Seed script: loads data-generate.csv and data-scraping.json
 * into Supabase PostgreSQL via Prisma.
 *
 * Usage:
 *   npx tsx demos/shared/prisma/seed.ts
 */

import { PrismaClient } from "@prisma/client";
import * as fs from "fs";
import * as path from "path";

const prisma = new PrismaClient();

// ─── CSV parser (no external dependency) ─────────────────────────

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

function safeParseJSON(val: string): unknown {
  if (!val || val.trim() === "" || val === "nan") return [];
  try {
    return JSON.parse(val.replace(/'/g, '"'));
  } catch {
    try {
      return JSON.parse(val);
    } catch {
      return [];
    }
  }
}

function safeFloat(val: string | undefined): number | null {
  if (!val || val.trim() === "" || val === "nan") return null;
  const n = parseFloat(val);
  return isNaN(n) ? null : n;
}

function safeStr(val: string | undefined): string | null {
  if (!val || val.trim() === "" || val === "nan") return null;
  return val.trim();
}

// ─── Load Data_Generate CSV ──────────────────────────────────────

async function seedDataGenerate() {
  const csvPath = path.resolve(__dirname, "../../../data-generate.csv");
  if (!fs.existsSync(csvPath)) {
    console.log("⚠ data-generate.csv not found, skipping");
    return;
  }

  console.log("📂 Loading data-generate.csv...");
  const content = fs.readFileSync(csvPath, "utf-8");
  const lines = content.split("\n").filter((l) => l.trim());
  const headers = parseCSVLine(lines[0]);

  const col = (row: string[], name: string) => {
    const idx = headers.indexOf(name);
    return idx >= 0 ? row[idx] : undefined;
  };

  let count = 0;
  const batchSize = 500;
  const batch: Parameters<typeof prisma.studentAnswer.create>[0]["data"][] = [];

  for (let i = 1; i < lines.length; i++) {
    const row = parseCSVLine(lines[i]);
    if (row.length < 5) continue;

    const pertType = safeStr(col(row, "perturbation_type"));

    batch.push({
      sampleId: `GEN_${String(i).padStart(5, "0")}`,
      sourceDataset: "data_generate",
      originalId: col(row, "instance_id") || `gen_${i}`,
      questionId: col(row, "question_id") || "",
      domain: col(row, "domain") || "",
      subdomain: col(row, "subdomain") || "",
      difficulty: col(row, "difficulty") || "medium",
      question: col(row, "question") || "",
      referenceAnswer: col(row, "reference_answer") || "",
      studentAnswer: col(row, "student_answer") || "",
      scoreRaw: safeFloat(col(row, "semantic_correctness_score_0_5")),
      label5way: safeStr(col(row, "label_5way")),
      label3way: safeStr(col(row, "label_3way")),
      label2way: safeStr(col(row, "label_2way")),
      keyConcepts: safeParseJSON(col(row, "key_concepts") || "[]"),
      misconceptionTags: safeParseJSON(col(row, "misconception_tags") || "[]"),
      missingConcepts: safeParseJSON(col(row, "missing_concepts") || "[]"),
      extraIncorrectClaims: safeParseJSON(col(row, "extra_incorrect_claims") || "[]"),
      feedbackShort: safeStr(col(row, "feedback_short")),
      feedbackDetailed: safeStr(col(row, "feedback_detailed")),
      feedbackType: safeStr(col(row, "feedback_type")),
      feedbackTone: safeStr(col(row, "feedback_tone")),
      split: col(row, "split") || "",
      isSynthetic: true,
      isHumanAnnotated: false,
      isAdversarial: !!pertType,
      perturbationType: pertType,
      adversarialVariantOf: safeStr(col(row, "adversarial_variant_of")),
      studentAnswerStyle: safeStr(col(row, "student_answer_style")),
      annotationConfidence: safeFloat(col(row, "annotation_confidence")),
      usableForGrading: true,
      usableForFeedback: true,
      usableForMisconceptionMining: true,
      usableForRobustnessEval: true,
    });

    if (batch.length >= batchSize) {
      await prisma.studentAnswer.createMany({ data: batch, skipDuplicates: true });
      count += batch.length;
      console.log(`  ✓ ${count} records inserted...`);
      batch.length = 0;
    }
  }

  if (batch.length > 0) {
    await prisma.studentAnswer.createMany({ data: batch, skipDuplicates: true });
    count += batch.length;
  }

  console.log(`✅ Data_Generate: ${count} records seeded`);
}

// ─── Load Data_Scraping JSON ─────────────────────────────────────

async function seedDataScraping() {
  const jsonPath = path.resolve(__dirname, "../../../data-scraping.json");
  if (!fs.existsSync(jsonPath)) {
    console.log("⚠ data-scraping.json not found, skipping");
    return;
  }

  console.log("📂 Loading data-scraping.json...");
  const raw = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

  const batch = raw.map((entry: Record<string, string>, idx: number) => ({
    sampleId: `SCR_${String(idx + 1).padStart(5, "0")}`,
    sourceDataset: "data_scraping",
    originalId: entry.id || `scr_${idx}`,
    questionId: entry.id || `scr_${idx}`,
    domain: (entry.label || "unknown").replace("openstax_", ""),
    subdomain: "general",
    difficulty: "unknown",
    question: entry.questions || "",
    referenceAnswer: entry.reference_answer || "",
    studentAnswer: entry.student_answer || "",
    isSynthetic: false,
    isHumanAnnotated: false,
    usableForGrading: false,
    usableForFeedback: false,
    usableForMisconceptionMining: false,
    usableForRobustnessEval: false,
  }));

  await prisma.studentAnswer.createMany({ data: batch, skipDuplicates: true });
  console.log(`✅ Data_Scraping: ${batch.length} records seeded`);
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  console.log("🌱 Seeding ASAG database...\n");

  // Clear existing data
  console.log("🗑  Clearing existing data...");
  await prisma.feedbackResult.deleteMany();
  await prisma.clusterAssignment.deleteMany();
  await prisma.cluster.deleteMany();
  await prisma.gradingResult.deleteMany();
  await prisma.studentAnswer.deleteMany();

  await seedDataGenerate();
  await seedDataScraping();

  const total = await prisma.studentAnswer.count();
  console.log(`\n🎉 Done! Total records in database: ${total}`);
}

main()
  .catch((e) => {
    console.error("❌ Seed failed:", e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
