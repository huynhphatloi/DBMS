import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/clusters — Fetch cluster data from database.
 *
 * Query params:
 *   strategy — embedding strategy (answer_only, question_answer, full_triplet)
 *   method   — clustering method (kmeans, hdbscan, bertopic)
 */
export async function GET(req: NextRequest) {
  try {
    const { prisma } = await import("@/lib/db");

    const url = req.nextUrl;
    const strategy = url.searchParams.get("strategy") || "question_answer";
    const method = url.searchParams.get("method") || "hdbscan";

    // Get clusters metadata
    const clusters = await prisma.cluster.findMany({
      where: { embeddingStrategy: strategy, clusteringMethod: method },
      orderBy: { clusterId: "asc" },
    });

    // Get assignments with student answer data
    const assignments = await prisma.clusterAssignment.findMany({
      where: { embeddingStrategy: strategy, clusteringMethod: method },
      include: {
        studentAnswer: {
          select: {
            sampleId: true,
            question: true,
            studentAnswer: true,
            label5way: true,
            scoreRaw: true,
            domain: true,
            questionId: true,
            misconceptionTags: true,
          },
        },
      },
    });

    return NextResponse.json({
      clusters,
      assignments,
      strategy,
      method,
      totalAssignments: assignments.length,
    });
  } catch {
    return NextResponse.json({
      clusters: [],
      assignments: [],
      message: "Database not connected. Using mock data.",
    });
  }
}
