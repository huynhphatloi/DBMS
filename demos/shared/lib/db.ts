/**
 * Shared Prisma client singleton.
 * Prevents multiple instances in Next.js dev mode (hot reload).
 *
 * Usage in any demo app:
 *   import { prisma } from "@/lib/db";
 */

import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
