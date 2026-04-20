import { NextRequest, NextResponse } from "next/server";

function tokenize(t: string) { return t.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean); }

const TONE_OPENERS: Record<string, string> = {
  friendly: "Hey! Good effort on this one. Let me break it down for you:",
  academic: "The submitted response demonstrates partial understanding of the topic. Below is a detailed analysis:",
  strict: "Assessment complete. The answer requires significant improvement. Details below:",
};

export async function POST(req: NextRequest) {
  const { question, studentAnswer, tone } = (await req.json()) as {
    question: string; studentAnswer: string; tone: string;
  };

  await new Promise((r) => setTimeout(r, 1000 + Math.random() * 800));

  const qTokens = tokenize(question);
  const sTokens = tokenize(studentAnswer);
  const sSet = new Set(sTokens);

  // Mock concept extraction from question
  const concepts = [...new Set(qTokens.filter((w) => w.length > 4))].slice(0, 8);
  const mentioned = concepts.filter((c) => sSet.has(c));
  const missed = concepts.filter((c) => !sSet.has(c));

  const strengths: string[] = [];
  if (sTokens.length > 5) strengths.push("Your answer has reasonable length and attempts to address the question");
  if (mentioned.length > 0) strengths.push(`You correctly referenced: ${mentioned.join(", ")}`);
  if (sTokens.length > 15) strengths.push("Good level of detail in your explanation");
  if (strengths.length === 0) strengths.push("You attempted to answer the question");

  const weaknesses: string[] = [];
  if (missed.length > 0) weaknesses.push(`Missing key concepts: ${missed.join(", ")}`);
  if (sTokens.length < 10) weaknesses.push("Your answer is too brief — try to elaborate more");
  const informal = ["stuff", "things", "like", "kinda", "gonna", "wanna"];
  const usedInformal = informal.filter((w) => sSet.has(w));
  if (usedInformal.length > 0) weaknesses.push(`Informal language detected: "${usedInformal.join('", "')}". Use scientific terminology instead`);
  if (weaknesses.length === 0) weaknesses.push("Minor: could add more specific examples");

  const suggestions: string[] = [];
  if (missed.length > 0) suggestions.push(`Include these concepts in your answer: ${missed.join(", ")}`);
  suggestions.push("Use specific terminology from your textbook or lecture notes");
  if (sTokens.length < 15) suggestions.push("Expand your answer to 3-4 sentences minimum");
  suggestions.push("Try to explain the 'why' behind each point, not just the 'what'");

  const completeness = Math.min(100, Math.round((mentioned.length / Math.max(concepts.length, 1)) * 100));
  const accuracy = Math.min(100, Math.round(60 + Math.random() * 30));
  const terminology = sTokens.length > 10 && usedInformal.length === 0 ? Math.round(50 + Math.random() * 40) : Math.round(20 + Math.random() * 30);
  const overall = Math.round((completeness + accuracy + terminology) / 3);

  return NextResponse.json({
    openingMessage: TONE_OPENERS[tone] || TONE_OPENERS.friendly,
    strengths,
    weaknesses,
    suggestions,
    scores: [
      { label: "Completeness", value: completeness },
      { label: "Accuracy", value: accuracy },
      { label: "Terminology", value: terminology },
      { label: "Overall", value: overall },
    ],
  });
}
