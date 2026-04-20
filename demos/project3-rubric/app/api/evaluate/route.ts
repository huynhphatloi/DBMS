import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { question, criteria, studentAnswer } = body as {
    question: string;
    criteria: { name: string; weight: number; description: string }[];
    studentAnswer: string;
  };

  await new Promise((r) => setTimeout(r, 900 + Math.random() * 500));

  const lower = studentAnswer.toLowerCase();
  const scores = criteria.map((c) => {
    const descWords = c.description.toLowerCase().split(/\s+/).filter((w) => w.length > 3);
    const matched = descWords.filter((w) => lower.includes(w));
    const ratio = descWords.length > 0 ? matched.length / descWords.length : 0.5;
    const raw = Math.min(10, Math.round((ratio * 7 + Math.random() * 3) * 10) / 10);
    return {
      criterion: c.name,
      weight: c.weight,
      score: raw,
      maxScore: 10,
      matchedTerms: matched.slice(0, 5),
      explanation:
        raw >= 7
          ? `Good coverage of ${c.name.toLowerCase()}. Key terms addressed.`
          : raw >= 4
          ? `Partial coverage of ${c.name.toLowerCase()}. Some key terms missing.`
          : `Weak coverage of ${c.name.toLowerCase()}. Most key terms not addressed.`,
    };
  });

  const totalWeight = criteria.reduce((s, c) => s + c.weight, 0);
  const weightedSum = scores.reduce((s, sc) => s + (sc.score * sc.weight) / totalWeight, 0);
  const finalScore = Math.round(weightedSum * 10) / 10;

  const grade =
    finalScore >= 9 ? "A" : finalScore >= 8 ? "A-" : finalScore >= 7 ? "B+" :
    finalScore >= 6 ? "B" : finalScore >= 5 ? "B-" : finalScore >= 4 ? "C" : "D";

  return NextResponse.json({ scores, finalScore, maxScore: 10, grade });
}
