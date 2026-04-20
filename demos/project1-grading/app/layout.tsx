import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASAG Grader — Teacher's Grading Dashboard",
  description: "Automatic Short Answer Grading Demo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans text-gray-800 antialiased">{children}</body>
    </html>
  );
}
