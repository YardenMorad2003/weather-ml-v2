import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Weather ML",
  description: "Vibe-based city climate recommender",
};

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#09090b",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-zinc-950 text-zinc-100">
        <div aria-hidden className="aurora" />
        <div aria-hidden className="aurora-grain" />
        <nav className="border-b border-zinc-800 bg-zinc-950/60 backdrop-blur sticky top-0 z-10">
          <div className="mx-auto max-w-5xl px-6 py-4 flex items-center gap-6">
            <span className="font-semibold tracking-tight">weather-ml</span>
            <div className="flex gap-4 text-sm text-zinc-400">
              <Link href="/" className="hover:text-zinc-100 transition">Query</Link>
              <Link href="/tournament" className="hover:text-zinc-100 transition">Pick</Link>
              <Link href="/explorer" className="hover:text-zinc-100 transition">Explorer</Link>
              <Link href="/about" className="hover:text-zinc-100 transition">How it works</Link>
            </div>
          </div>
        </nav>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
