import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/history — Fetch past feedback results.
 * POST /api/history — Save a new feedback result.
 */
export async function GET(req: NextRequest) {
  try {
    const { prisma } = await import("@/lib/db");

    const limit = parseInt(req.nextUrl.searchParams.get("limit") || "10");

    const results = await prisma.feedbackResult.findMany({
      take: limit,
      orderBy: { createdAt: "desc" },
    });

    return NextResponse.json({ results });
  } catch {
    return NextResponse.json({ results: [], message: "Database not connected" });
  }
}

export async function POST(req: NextRequest) {
  try {
    const { prisma } = await import("@/lib/db");
    const body = await req.json();

    const result = await prisma.feedbackResult.create({
      data: {
        question: body.question,
        referenceAnswer: body.referenceAnswer || "",
        studentAnswer: body.studentAnswer,
        predictedLabel: body.predictedLabel || "unknown",
        tone: body.tone || "friendly",
        feedbackShort: body.feedbackShort || "",
        feedbackDetailed: body.feedbackDetailed || "",
        strategy: body.strategy || "template",
        consistencyScore: body.consistencyScore,
        usedFallback: body.usedFallback || false,
        presentConcepts: body.presentConcepts || [],
        missingConcepts: body.missingConcepts || [],
        contradictedConcepts: body.contradictedConcepts || [],
        completenessScore: body.scores?.completeness,
        accuracyScore: body.scores?.accuracy,
        terminologyScore: body.scores?.terminology,
        overallScore: body.scores?.overall,
      },
    });

    return NextResponse.json({ id: result.id, saved: true });
  } catch {
    return NextResponse.json({ saved: false, message: "Database not connected" });
  }
}
