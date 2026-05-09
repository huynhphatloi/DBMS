// @ts-nocheck
import { NextResponse } from "next/server";

/**
 * GET /api/stats — Database statistics for the dashboard header.
 * Returns empty stats if database is not connected.
 */
export async function GET() {
  try {
    const { PrismaClient } = require("@prisma/client");
    const prisma = new PrismaClient();

    const [total, bySource, byLabel] = await Promise.all([
      prisma.studentAnswer.count(),
      prisma.studentAnswer.groupBy({
        by: ["sourceDataset"],
        _count: true,
      }),
      prisma.studentAnswer.groupBy({
        by: ["label3way"],
        _count: true,
        where: { label3way: { not: null } },
      }),
    ]);

    await prisma.$disconnect();

    return NextResponse.json({
      total,
      bySource: Object.fromEntries(
        bySource.map((g) => [g.sourceDataset, g._count])
      ),
      byLabel: Object.fromEntries(
        byLabel.map((g) => [g.label3way || "null", g._count])
      ),
      connected: true,
    });
  } catch {
    return NextResponse.json({
      total: 0,
      bySource: {},
      byLabel: {},
      connected: false,
      message: "Database not connected. Set DATABASE_URL in .env.local",
    });
  }
}
