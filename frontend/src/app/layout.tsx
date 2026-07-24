import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KIRAG — Medicolegal Document Processing & RAG Workstation",
  description: "Enterprise medicolegal document audit, layout inspection, RAG query engine, and case timeline workstation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full">
      <body className="bg-[#0b0f19] text-slate-100 min-h-screen antialiased flex">
        {children}
      </body>
    </html>
  );
}
