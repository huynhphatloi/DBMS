import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "StudyBuddy — AI Learning Assistant",
  description: "Feedback Generation System Demo",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans text-indigo-950 antialiased">{children}</body>
    </html>
  );
}
