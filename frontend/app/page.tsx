"use client";

import { useEffect, useState } from "react";
import {
  recommendText,
  type AnchorError,
  type CityResult,
  type RecommendResponse,
  type Vibe,
} from "@/lib/api";
import {
  preloadContrastive,
  rankContrastive,
  type ContrastiveResult,
} from "@/lib/contrastive";
import { CityCard } from "./components/CityCard";

const EXAMPLES = [
  "Tokyo but less humid summers",
  "London but warmer winters",
  "Seattle but sunnier",
  "warm coastal sunny",
  "almost always sunny non-raining",
];

type Mode = "auto" | "classical" | "smart";

function asCityResult(r: ContrastiveResult): CityResult {
  // CityCard re-fetches country/lat/lon/stats via getCity when expanded, so
  // empty placeholders are fine here. The contrastive ranker doesn't produce
  // per-feature reasons; expanded view will just omit that block.
  return {
    city: r.city,
    country: "",
    lat: 0,
    lon: 0,
    similarity: r.similarity,
    reasons: [],
  };
}

export default function QueryPage() {
  const [mode, setMode] = useState<Mode>("auto");
  const [compare, setCompare] = useState(false);
  const [text, setText] = useState("New York but warmer winters");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [classicalResp, setClassicalResp] = useState<RecommendResponse | null>(null);
  const [smartResp, setSmartResp] = useState<ContrastiveResult[] | null>(null);
  const [routedTo, setRoutedTo] = useState<"classical" | "smart" | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  // Warm the MiniLM weights + manifest in the background so the first Smart
  // query feels close to instant. No-op if already loaded.
  useEffect(() => {
    preloadContrastive();
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      setCompare(params.get("compare") === "1");
    }
  }, []);

  async function runQuery(q: string) {
    setText(q);
    setLoading(true);
    setError(null);
    setExpanded(null);
    setRoutedTo(null);
    try {
      if (compare) {
        const [c, s] = await Promise.all([
          recommendText(q),
          rankContrastive(q, 10),
        ]);
        setClassicalResp(c);
        setSmartResp(s);
      } else if (mode === "classical") {
        const c = await recommendText(q);
        setClassicalResp(c);
        setSmartResp(null);
        setRoutedTo("classical");
      } else if (mode === "smart") {
        const s = await rankContrastive(q, 10);
        setSmartResp(s);
        setClassicalResp(null);
        setRoutedTo("smart");
      } else {
        // Auto: parse first via classical (it includes parser output), then
        // route on whether the parser extracted anything structured. The σ-eval
        // showed classical wins on any query the vocabulary covers — anchor
        // OR vibes — and contrastive's value is the residual (queries the
        // vocabulary can't represent at all).
        const c = await recommendText(q);
        if (c.parsed.anchor_city || c.parsed.vibes.length > 0) {
          setClassicalResp(c);
          setSmartResp(null);
          setRoutedTo("classical");
        } else {
          const s = await rankContrastive(q, 10);
          setClassicalResp(null);
          setSmartResp(s);
          setRoutedTo("smart");
        }
      }
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
        <h1 className="text-3xl font-semibold tracking-tight">
          Describe your ideal climate
        </h1>
        <p className="mt-2 text-zinc-400">
          Name a city, tweak the vibe. e.g. &quot;Tokyo but less humid
          summers&quot;.
        </p>
      </header>

      {!compare && (
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs text-zinc-500">Ranker:</span>
          <div className="inline-flex rounded-lg border border-zinc-800 bg-zinc-900/40 p-0.5 text-xs">
            {(["auto", "classical", "smart"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`rounded-md px-3 py-1.5 transition capitalize ${
                  mode === m
                    ? "bg-zinc-100 text-zinc-950"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {m === "smart" ? "Smart (Contrastive)" : m}
              </button>
            ))}
          </div>
          <span className="text-[11px] text-zinc-500">
            {mode === "auto" &&
              "Structured queries (anchor or vibe) → classical. Pure free-form → contrastive."}
            {mode === "smart" &&
              "Runs in your browser. First query downloads ~13 MB once."}
          </span>
        </div>
      )}

      {compare && (
        <div className="rounded-lg border border-amber-900 bg-amber-950/30 p-3 text-xs text-amber-200">
          Compare mode (<code>?compare=1</code>) — both rankers run on every
          query. Highlighted cities appear in both top-10 lists.
        </div>
      )}

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

      {compare && classicalResp && smartResp && (
        <CompareView classical={classicalResp} smart={smartResp} />
      )}

      {!compare && routedTo && (
        <RouteBadge
          routedTo={routedTo}
          mode={mode}
          parsedAnchor={classicalResp?.parsed.anchor_city ?? null}
        />
      )}

      {!compare && routedTo === "classical" && classicalResp && (
        <ClassicalResults
          resp={classicalResp}
          expanded={expanded}
          setExpanded={setExpanded}
          onSuggestion={runQuery}
        />
      )}

      {!compare && routedTo === "smart" && smartResp && (
        <SmartResults
          results={smartResp}
          expanded={expanded}
          setExpanded={setExpanded}
        />
      )}
    </div>
  );
}

function RouteBadge({
  routedTo,
  mode,
  parsedAnchor,
}: {
  routedTo: "classical" | "smart";
  mode: Mode;
  parsedAnchor: string | null;
}) {
  if (mode !== "auto") {
    return (
      <div className="text-xs text-zinc-500">
        Ranker: <span className="text-zinc-300">{routedTo}</span> (manual
        override)
      </div>
    );
  }
  return (
    <div className="text-xs text-zinc-500 flex items-center gap-2 flex-wrap">
      <span>Auto-routed to</span>
      <span
        className={
          routedTo === "classical"
            ? "rounded bg-zinc-800 text-zinc-100 px-2 py-0.5"
            : "rounded bg-indigo-950 text-indigo-200 border border-indigo-900 px-2 py-0.5"
        }
      >
        {routedTo === "classical"
          ? "classical (vocabulary-covered)"
          : "contrastive (free-form)"}
      </span>
      <span className="text-zinc-600">·</span>
      <span>
        {parsedAnchor
          ? `anchor "${parsedAnchor}" extracted`
          : "no anchor extracted"}
      </span>
    </div>
  );
}

function ClassicalResults({
  resp,
  expanded,
  setExpanded,
  onSuggestion,
}: {
  resp: RecommendResponse;
  expanded: string | null;
  setExpanded: (v: string | null) => void;
  onSuggestion: (q: string) => void;
}) {
  const hasError = resp.anchor_error !== null;
  return (
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
            <span
              key={i}
              className="rounded bg-indigo-950 text-indigo-300 border border-indigo-900 px-2 py-1"
            >
              {v.intensity} {v.axis} · {v.scope}
            </span>
          ))}
          {resp.parsed.vibes.length === 0 && (
            <span className="text-zinc-500">no vibes extracted</span>
          )}
        </div>
      </div>

      {resp.anchor_error && (
        <AnchorErrorNotice
          err={resp.anchor_error}
          parsedVibes={resp.parsed.vibes}
          onSuggestion={onSuggestion}
        />
      )}

      {!hasError && (
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
      )}

      {!hasError && (
        <div className="grid gap-3">
          {resp.results.map((c, i) => (
            <CityCard
              key={c.city}
              rank={i + 1}
              result={c}
              expanded={expanded === c.city}
              onToggle={() => setExpanded(expanded === c.city ? null : c.city)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function AnchorErrorNotice({
  err,
  parsedVibes,
  onSuggestion,
}: {
  err: AnchorError;
  parsedVibes: Vibe[];
  onSuggestion: (q: string) => void;
}) {
  const vibeText = parsedVibes
    .map((v) => `${v.intensity} ${v.axis} ${v.scope}`)
    .join(" ")
    .replaceAll("_", " ");
  return (
    <div className="rounded-lg border border-amber-900 bg-amber-950/30 p-4 text-sm space-y-3">
      <div className="text-amber-200">
        Couldn&apos;t find <span className="font-medium">&quot;{err.input}&quot;</span>{" "}
        on the map. Nothing in the canonical 230 cities or the on-demand
        geocoder matched it.
      </div>
      {err.suggestions.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-amber-300/70">Did you mean:</span>
          {err.suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() =>
                onSuggestion(vibeText ? `${s} but ${vibeText}` : s)
              }
              className="rounded-full border border-amber-800 bg-amber-950/50 px-3 py-1 text-xs text-amber-100 hover:bg-amber-900/60 hover:border-amber-700 transition"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      {parsedVibes.length > 0 && (
        <div className="text-xs text-amber-300/70">
          Or{" "}
          <button
            type="button"
            onClick={() => onSuggestion(vibeText)}
            className="underline hover:text-amber-100"
          >
            search just the vibe
          </button>{" "}
          ({vibeText.trim() || "no vibe extracted"}) using the contrastive
          ranker.
        </div>
      )}
    </div>
  );
}

function SmartResults({
  results,
  expanded,
  setExpanded,
}: {
  results: ContrastiveResult[];
  expanded: string | null;
  setExpanded: (v: string | null) => void;
}) {
  return (
    <section className="space-y-4">
      <div className="rounded-lg border border-indigo-900 bg-indigo-950/30 p-3 text-xs text-indigo-200">
        Ranked by the dual-encoder contrastive model trained on synthetic
        (query → city) triples. No anchor extraction — the model reads
        free-form text directly.
      </div>
      <ContrastiveScoreLegend results={results} />
      <div className="grid gap-3">
        {results.map((r) => (
          <CityCard
            key={r.city}
            rank={r.rank + 1}
            result={asCityResult(r)}
            expanded={expanded === r.city}
            onToggle={() => setExpanded(expanded === r.city ? null : r.city)}
          />
        ))}
      </div>
    </section>
  );
}

type Confidence = {
  label: "confident" | "moderate" | "uncertain";
  accent: string;
  hint: string;
};

function classifySpread(gapPts: number): Confidence {
  if (gapPts >= 15) {
    return {
      label: "confident",
      accent:
        "text-emerald-300 bg-emerald-950/40 border-emerald-900",
      hint: "wide gap — model has a strong opinion on this query",
    };
  }
  if (gapPts >= 8) {
    return {
      label: "moderate",
      accent: "text-amber-300 bg-amber-950/40 border-amber-900",
      hint: "modest gap — the middle of the list is hard to distinguish",
    };
  }
  return {
    label: "uncertain",
    accent: "text-zinc-400 bg-zinc-900/40 border-zinc-800",
    hint: "narrow gap — model is unsure; treat as low-confidence",
  };
}

function ContrastiveScoreLegend({ results }: { results: ContrastiveResult[] }) {
  if (results.length === 0) return null;
  const top = results[0].similarity * 100;
  const bottom = results[results.length - 1].similarity * 100;
  const gap = top - bottom;
  const c = classifySpread(gap);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/30 p-3 text-xs space-y-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">
        How to read these scores
      </div>
      <p className="text-zinc-400 leading-relaxed">
        Cosine similarity in the model&apos;s learned 32-d space.{" "}
        <span className="text-zinc-300">Not</span>{" "}
        the same scale as classical scores — there&apos;s no fixed threshold
        for &quot;good.&quot; What matters is the{" "}
        <span className="text-zinc-300">gap</span>{" "}
        between the top and bottom of the list.
      </p>
      <div className="flex items-center gap-2 flex-wrap text-zinc-400">
        <span>Spread #1→#{results.length}:</span>
        <span className="tabular-nums text-zinc-200">
          {top.toFixed(1)}% → {bottom.toFixed(1)}% ({gap.toFixed(1)} pts)
        </span>
        <span className={`rounded border px-2 py-0.5 ${c.accent}`}>
          {c.label}
        </span>
        <span className="text-zinc-500 hidden sm:inline">— {c.hint}</span>
      </div>
    </div>
  );
}

function CompareView({
  classical,
  smart,
}: {
  classical: RecommendResponse;
  smart: ContrastiveResult[];
}) {
  const classicalCities = new Set(classical.results.map((r) => r.city));
  const smartCities = new Set(smart.map((r) => r.city));
  const overlap = new Set(
    [...classicalCities].filter((c) => smartCities.has(c))
  );

  return (
    <section className="space-y-3">
      <div className="text-xs text-zinc-500">
        Overlap: {overlap.size} of 10 cities appear in both rankings.
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <RankColumn
          title="Classical (anchor + tweak)"
          subtitle={
            classical.parsed.anchor_city
              ? `anchor: ${classical.parsed.anchor_city}`
              : "no anchor — falls back to vibes only"
          }
          rows={classical.results.map((r) => ({
            city: r.city,
            similarity: r.similarity,
          }))}
          overlap={overlap}
          accent="zinc"
          confidence={null}
        />
        <RankColumn
          title="Contrastive (browser)"
          subtitle="dual-encoder, free-form text"
          rows={smart.map((r) => ({
            city: r.city,
            similarity: r.similarity,
          }))}
          overlap={overlap}
          accent="indigo"
          confidence={(() => {
            if (smart.length === 0) return null;
            const gap = (smart[0].similarity - smart[smart.length - 1].similarity) * 100;
            return classifySpread(gap);
          })()}
        />
      </div>
    </section>
  );
}

function RankColumn({
  title,
  subtitle,
  rows,
  overlap,
  accent,
  confidence,
}: {
  title: string;
  subtitle: string;
  rows: { city: string; similarity: number }[];
  overlap: Set<string>;
  accent: "zinc" | "indigo";
  confidence: Confidence | null;
}) {
  const headerBg =
    accent === "indigo"
      ? "border-indigo-900 bg-indigo-950/30"
      : "border-zinc-800 bg-zinc-900/40";
  return (
    <div className={`rounded-lg border ${headerBg} overflow-hidden`}>
      <div className="px-4 py-3 border-b border-zinc-800 flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-medium text-zinc-100">{title}</div>
          <div className="text-[11px] text-zinc-500">{subtitle}</div>
        </div>
        {confidence && (
          <span
            className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wide ${confidence.accent}`}
            title={confidence.hint}
          >
            {confidence.label}
          </span>
        )}
      </div>
      <ol className="divide-y divide-zinc-800/60">
        {rows.map((r, i) => {
          const isOverlap = overlap.has(r.city);
          return (
            <li
              key={r.city}
              className={`flex items-center justify-between px-4 py-2.5 text-sm ${
                isOverlap ? "bg-emerald-950/20" : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="text-zinc-500 tabular-nums w-6">{i + 1}</span>
                <span
                  className={
                    isOverlap ? "text-emerald-200" : "text-zinc-200"
                  }
                >
                  {r.city}
                </span>
              </div>
              <span className="text-zinc-400 tabular-nums text-xs">
                {(r.similarity * 100).toFixed(1)}%
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
