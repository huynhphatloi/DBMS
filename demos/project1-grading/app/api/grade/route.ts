import { NextRequest, NextResponse } from "next/server";

/* ---------- mock grading logic (replace with real FastAPI call) ---------- */

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);
}

function jaccard(a: string[], b: string[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  const inter = [...setA].filter((x) => setB.has(x)).length;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : inter / union;
}

function overlapRatio(ref: string[], stu: string[]): number {
  const refSet = new Set(ref);
  const matched = stu.filter((w) => refSet.has(w)).length;
  return ref.length === 0 ? 0 : matched / ref.length;
}

interface Span {
  text: string;
  match: "matched" | "partial" | "missing" | "none";
}

function buildSpans(reference: string, student: string): { refSpans: Span[]; stuSpans: Span[] } {
  const refTokens = tokenize(reference);
  const stuTokens = tokenize(student);
  const stuSet = new Set(stuTokens);
  const refSet = new Set(refTokens);

  const refWords = reference.split(/(\s+)/);
  const stuWords = student.split(/(\s+)/);

  const refSpans: Span[] = refWords.map((w) => {
    if (/^\s+$/.test(w)) return { text: w, match: "none" };
    const clean = w.toLowerCase().replace(/[^\w]/g, "");
    if (stuSet.has(clean)) return { text: w, match: "matched" };
    return { text: w, match: "missing" };
  });

  const stuSpans: Span[] = stuWords.map((w) => {
    if (/^\s+$/.test(w)) return { text: w, match: "none" };
    const clean = w.toLowerCase().replace(/[^\w]/g, "");
    if (refSet.has(clean)) return { text: w, match: "matched" };
    return { text: w, match: "partial" };
  });

  return { refSpans, stuSpans };
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { question, referenceAnswer, studentAnswer } = body as {
    question: string;
    referenceAnswer: string;
    studentAnswer: string;
  };

  // simulate latency
  await new Promise((r) => setTimeout(r, 800 + Math.random() * 600));

  const refTokens = tokenize(referenceAnswer);
  const stuTokens = tokenize(studentAnswer);

  const semantic = Math.min(1, jaccard(refTokens, stuTokens) + 0.15 + Math.random() * 0.1);
  const lexical = jaccard(refTokens, stuTokens);
  const keyConcept = overlapRatio(refTokens, stuTokens);
  const overall = semantic * 0.5 + lexical * 0.2 + keyConcept * 0.3;

  const score = Math.round(overall * 100) / 10; // 0-10
  const confidence = Math.round((0.7 + Math.random() * 0.25) * 100);
  const label: "correct" | "partial" | "incorrect" =
    score >= 7 ? "correct" : score >= 4 ? "partial" : "incorrect";

  // key concepts
  const importantRef = [...new Set(refTokens)].filter((t) => t.length > 3).slice(0, 8);
  const stuSet = new Set(stuTokens);
  const matched = importantRef.filter((c) => stuSet.has(c));
  const missing = importantRef.filter((c) => !stuSet.has(c));

  const { refSpans, stuSpans } = buildSpans(referenceAnswer, studentAnswer);

  const explanationParts: string[] = [];
  if (matched.length > 0)
    explanationParts.push(`The student correctly addressed: ${matched.join(", ")}.`);
  if (missing.length > 0)
    explanationParts.push(`However, the answer is missing key concepts: ${missing.join(", ")}.`);
  if (score >= 7) explanationParts.push("Overall, this is a strong answer.");
  else if (score >= 4) explanationParts.push("The answer shows partial understanding but needs more detail.");
  else explanationParts.push("The answer does not adequately address the question.");

  return NextResponse.json({
    score,
    confidence,
    label,
    similarity: {
      overall: Math.round(overall * 100) / 100,
      semantic: Math.round(semantic * 100) / 100,
      lexical: Math.round(lexical * 100) / 100,
      keyConcept: Math.round(keyConcept * 100) / 100,
    },
    explanation: explanationParts.join(" "),
    concepts: { matched, missing },
    refSpans,
    stuSpans,
  });
}
