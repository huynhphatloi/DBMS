import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "MisconceptionMiner — Error Pattern Analytics",
  description: "Misconception & Error Pattern Mining Demo",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
