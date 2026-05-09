// @ts-nocheck
import { NextRequest, NextResponse } from "next/server";

/**
 * GET /api/answers — Fetch student answers from the database.
 * Returns empty if database is not connected.
 */
export async function GET(req: NextRequest) {
  try {
    const { PrismaClient } = require("@prisma/client");
    const prisma = new PrismaClient();

    const url = req.nextUrl;
    const source = url.searchParams.get("source");
    const split = url.searchParams.get("split");
    const label = url.searchParams.get("label");
    const limit = Math.min(100, parseInt(url.searchParams.get("limit") || "20"));
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const search = url.searchParams.get("search");

    const where: Record<string, unknown> = {};
    if (source) where.sourceDataset = source;
    if (split) where.split = split;
    if (label) where.label3way = label;
    if (search) {
      where.OR = [
        { question: { contains: search, mode: "insensitive" } },
        { studentAnswer: { contains: search, mode: "insensitive" } },
      ];
    }

    const [records, total] = await Promise.all([
      prisma.studentAnswer.findMany({
        where,
        take: limit,
        skip: offset,
        orderBy: { id: "asc" },
        select: {
          sampleId: true,
          sourceDataset: true,
          questionId: true,
          domain: true,
          question: true,
          referenceAnswer: true,
          studentAnswer: true,
          scoreRaw: true,
          label2way: true,
          label3way: true,
          label5way: true,
          keyConcepts: true,
          missingConcepts: true,
          feedbackShort: true,
          split: true,
        },
      }),
      prisma.studentAnswer.count({ where }),
    ]);

    return NextResponse.json({ records, total, limit, offset });
  } catch {
    return NextResponse.json({
      records: [],
      total: 0,
      limit: 20,
      offset: 0,
      message: "Database not connected. Set DATABASE_URL in .env.local",
    });
  }
}
