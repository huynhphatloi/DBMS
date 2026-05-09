#!/bin/bash
# Setup script: installs dependencies and links shared Prisma schema
# Usage: cd demos && bash setup.sh

set -e

echo "🔧 Setting up ASAG demo apps..."
echo ""

# 1. Install shared dependencies
echo "📦 Installing shared dependencies..."
cd shared
npm install
cd ..

# 2. For each demo app: install deps + copy Prisma schema + generate client
for project in project1-grading project2-misconception project3-rubric project4-feedback; do
  echo ""
  echo "📦 Setting up $project..."
  cd "$project"

  # Install npm deps
  npm install

  # Add Prisma client dependency
  npm install @prisma/client@^5.14.0 2>/dev/null || true

  # Copy shared Prisma schema
  mkdir -p prisma
  cp ../shared/prisma/schema.prisma prisma/schema.prisma

  # Copy shared db client
  mkdir -p lib
  cp ../shared/lib/db.ts lib/db.ts

  # Generate Prisma client (needs DATABASE_URL in .env)
  if [ -f .env ] || [ -f .env.local ]; then
    npx prisma generate 2>/dev/null || echo "  ⚠ Prisma generate skipped (set DATABASE_URL first)"
  else
    echo "  ⚠ No .env file — copy .env.example and set DATABASE_URL"
  fi

  cd ..
done

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Create a Supabase project at https://supabase.com"
echo "  2. Copy connection strings to each app's .env.local"
echo "  3. Run: cd shared && npm run db:setup"
echo "  4. Run: cd project1-grading && npm run dev"
