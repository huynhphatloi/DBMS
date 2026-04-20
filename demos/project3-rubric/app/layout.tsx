import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "RubricGrader — Explainable Rubric-Based Grading",
  description: "Rubric-based Explainable Grading Demo",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans text-slate-900 antialiased">{children}</body>
    </html>
  );
}
