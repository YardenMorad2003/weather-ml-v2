"use client";

import { useState } from "react";
import { recommendText, type RecommendResponse } from "@/lib/api";
import { CityCard } from "./components/CityCard";

const EXAMPLES = [
  "Tokyo but less humid summers",
  "London but warmer winters",
  "Seattle but sunnier",
  "Dubai but cooler",
  "Oslo but milder",
];

export default function QueryPage() {
  const [text, setText] = useState("New York but warmer winters");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<RecommendResponse | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  async function runQuery(q: string) {
    setText(q);
    setLoading(true);
    setError(null);
    try {
      const r = await recommendText(q);
      setResp(r);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    await runQuery(text);
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Describe your ideal climate</h1>
        <p className="mt-2 text-zinc-400">
          Name a city, tweak the vibe. e.g. &quot;Tokyo but less humid summers&quot;.
        </p>
      </header>

      <form onSubmit={submit} className="flex gap-3">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="flex-1 rounded-lg bg-zinc-900/70 backdrop-blur border border-zinc-800 px-4 py-3 text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
          placeholder="New York but warmer winters"
        />
        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="rounded-lg bg-zinc-100 text-zinc-950 px-5 py-3 font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-white transition"
        >
          {loading ? "..." : "Find cities"}
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-zinc-500 self-center mr-1">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => runQuery(ex)}
            disabled={loading}
            className="rounded-full border border-zinc-800 bg-zinc-900/40 backdrop-blur px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-100 hover:border-zinc-600 hover:bg-zinc-900/80 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {ex}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {resp && (
        <section className="space-y-6">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 text-sm">
            <div className="text-zinc-400 mb-2">Parsed as</div>
            <div className="flex flex-wrap gap-2">
              {resp.parsed.anchor_city && (
                <span className="rounded bg-zinc-800 px-2 py-1 text-zinc-200">
                  anchor: {resp.parsed.anchor_city}
                </span>
              )}
              {resp.parsed.vibes.map((v, i) => (
                <span key={i} className="rounded bg-indigo-950 text-indigo-300 border border-indigo-900 px-2 py-1">
                  {v.intensity} {v.axis} · {v.scope}
                </span>
              ))}
              {resp.parsed.vibes.length === 0 && (
                <span className="text-zinc-500">no vibes extracted</span>
              )}
            </div>
          </div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-950/30 p-3 text-xs">
            <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-2">
              What the similarity scores mean
            </div>
            <div className="grid sm:grid-cols-2 gap-x-6 gap-y-1 text-zinc-400">
              <div className="flex items-baseline gap-2">
                <span className="text-emerald-400 tabular-nums w-14">80–90%+</span>
                <span>very close climatic twin</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-emerald-300 tabular-nums w-14">50–60%</span>
                <span>reasonable match</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-amber-400 tabular-nums w-14">30–40%</span>
                <span>best given your constraints, but diverges elsewhere</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-zinc-500 tabular-nums w-14">&lt;25%</span>
                <span>little in the dataset fits</span>
              </div>
            </div>
          </div>

          <div className="grid gap-3">
            {resp.results.map((c, i) => (
              <CityCard
                key={c.city}
                rank={i + 1}
                result={c}
                expanded={expanded === c.city}
                onToggle={() =>
                  setExpanded(expanded === c.city ? null : c.city)
                }
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
