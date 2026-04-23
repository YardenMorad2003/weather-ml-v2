"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getTournamentPair,
  getTournamentFinal,
  type PairResponse,
  type FinalResponse,
  type PairCity,
  type TournamentHistoryItem,
} from "@/lib/api";
import { CityCard } from "../components/CityCard";

const KB_VARIANTS = ["kb-a", "kb-b", "kb-c", "kb-d"] as const;

function kbClass(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return KB_VARIANTS[Math.abs(h) % KB_VARIANTS.length];
}

function newSeed(): number {
  return Math.floor(Math.random() * 1_000_000_000);
}

export default function TournamentPage() {
  const [history, setHistory] = useState<TournamentHistoryItem[]>([]);
  const [pair, setPair] = useState<PairResponse | null>(null);
  const [final, setFinal] = useState<FinalResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [seed, setSeed] = useState<number>(() => newSeed());
  // phase: "idle" (showing a pair), "animating" (post-click fade), "finalizing"
  const [phase, setPhase] = useState<"idle" | "animating" | "finalizing">("idle");
  const [bootProgress, setBootProgress] = useState(0);
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    getTournamentPair([], seed)
      .then((p) => setPair(p))
      .catch((e) => setError((e as Error).message))
      .finally(() => {
        setLoading(false);
        setBootProgress(100);
      });
    // seed is captured from initial state; we never want this effect to rerun
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Asymptotic progress bar during initial boot. Free-tier Render can
  // cold-start for 30-60s; we ease toward 95% over ~60s so the bar feels
  // responsive early but never falsely claims to be done.
  useEffect(() => {
    if (pair || final || error) return;
    const start = Date.now();
    const id = setInterval(() => {
      const elapsed = (Date.now() - start) / 1000;
      const pct = 100 * (1 - Math.exp(-elapsed / 22));
      setBootProgress(Math.min(95, pct));
    }, 150);
    return () => clearInterval(id);
  }, [pair, final, error]);

  const pick = useCallback(
    async (choice: PairCity, other: PairCity) => {
      if (!pair || phase !== "idle") return;
      const newHist: TournamentHistoryItem[] = [
        ...history,
        { shown: [choice.name, other.name], picked: choice.name },
      ];
      setPhase("animating");

      // brief fade so the swap doesn't feel jarring
      await new Promise((r) => setTimeout(r, 260));

      if (newHist.length >= pair.total_rounds) {
        setPhase("finalizing");
        try {
          const f = await getTournamentFinal(newHist);
          setHistory(newHist);
          setFinal(f);
          setPair(null);
        } catch (e) {
          setError((e as Error).message);
        } finally {
          setPhase("idle");
        }
        return;
      }

      setHistory(newHist);
      try {
        const next = await getTournamentPair(newHist, seed);
        setPair(next);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setPhase("idle");
      }
    },
    [history, pair, phase, seed]
  );

  useEffect(() => {
    if (!pair || phase !== "idle") return;
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (k === "arrowleft" || k === "a") {
        e.preventDefault();
        pick(pair.pair[0], pair.pair[1]);
      } else if (k === "arrowright" || k === "d") {
        e.preventDefault();
        pick(pair.pair[1], pair.pair[0]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pair, pick, phase]);

  function reset() {
    const nextSeed = newSeed();
    setSeed(nextSeed);
    setHistory([]);
    setFinal(null);
    setPair(null);
    setExpanded(null);
    setError(null);
    setLoading(true);
    setBootProgress(0);
    getTournamentPair([], nextSeed)
      .then((p) => setPair(p))
      .catch((e) => setError((e as Error).message))
      .finally(() => {
        setLoading(false);
        setBootProgress(100);
      });
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-3xl font-semibold tracking-tight">Pick</h1>
        <div className="rounded-lg border border-red-900 bg-red-950/40 p-4 text-red-300 text-sm">
          {error}
        </div>
        <button
          onClick={reset}
          className="rounded-lg bg-zinc-100 text-zinc-950 px-4 py-2 font-medium hover:bg-white transition"
        >
          Try again
        </button>
      </div>
    );
  }

  if (final) {
    return (
      <div className="space-y-8">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">Your climate</h1>
          <p className="mt-2 text-zinc-400">
            {`Based on your ${final.rounds_completed} picks, here's the top match and the next nine.`}
          </p>
        </header>

        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 text-sm">
          <div className="text-zinc-400 mb-2">You picked</div>
          <div className="flex flex-wrap gap-2">
            {final.picked.map((p, i) => (
              <span
                key={i}
                className="rounded bg-zinc-800 px-2 py-1 text-zinc-200"
              >
                {i + 1}. {p}
              </span>
            ))}
          </div>
        </div>

        <div className="grid gap-3">
          {final.results.map((c, i) => (
            <CityCard
              key={c.city}
              rank={i + 1}
              result={c}
              expanded={expanded === c.city}
              onToggle={() => setExpanded(expanded === c.city ? null : c.city)}
            />
          ))}
        </div>

        <button
          onClick={reset}
          className="rounded-lg bg-zinc-100 text-zinc-950 px-5 py-3 font-medium hover:bg-white transition"
        >
          Play again
        </button>
      </div>
    );
  }

  const round = pair?.round ?? 1;
  const total = pair?.total_rounds ?? 10;
  const pct = ((round - 1) / total) * 100;

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Pick the vibe</h1>
        <p className="mt-2 text-zinc-400">
          Two cities. Pick the one that feels more like your climate. Ten rounds,
          then we&apos;ll show you the full ranking.
        </p>
      </header>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-zinc-400">
          <span>Round {round} of {total}</span>
          <span className="text-zinc-500">← / → or A / D to pick</span>
        </div>
        <div className="h-1 rounded-full bg-zinc-800 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div
        className={`grid grid-cols-1 md:grid-cols-2 gap-1 rounded-2xl overflow-hidden border border-zinc-800 bg-zinc-950 transition-opacity duration-200 ${
          phase === "animating" ? "opacity-30" : "opacity-100"
        }`}
        style={{ minHeight: "min(70vh, 640px)" }}
      >
        {pair?.pair.map((city, idx) => (
          <PickCard
            key={`${round}-${idx}-${city.name}`}
            city={city}
            side={idx === 0 ? "left" : "right"}
            onPick={() => pick(city, pair.pair[idx === 0 ? 1 : 0])}
            disabled={phase !== "idle"}
          />
        ))}
        {!pair && loading && <BootLoader progress={bootProgress} />}
      </div>

      {phase === "finalizing" && (
        <div className="text-center text-sm text-zinc-500">
          Computing your climate…
        </div>
      )}
    </div>
  );
}

function PickCard({
  city,
  side,
  onPick,
  disabled,
}: {
  city: PairCity;
  side: "left" | "right";
  onPick: () => void;
  disabled: boolean;
}) {
  const kb = kbClass(city.name);
  const hasImage = Boolean(city.thumb_url);
  return (
    <button
      onClick={onPick}
      disabled={disabled}
      aria-label={`Pick ${city.name}`}
      className="relative group overflow-hidden cursor-pointer disabled:cursor-default"
    >
      <div className="absolute inset-0 overflow-hidden">
        {hasImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={city.thumb_url}
            alt={city.name}
            className={`w-full h-full object-cover ${kb}`}
            loading="eager"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-zinc-800 to-zinc-900" />
        )}
      </div>
      {/* bottom gradient so the city name is always readable */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/20 to-transparent pointer-events-none" />
      {/* subtle hover tint */}
      <div
        className={`absolute inset-0 transition-colors pointer-events-none ${
          disabled ? "" : "group-hover:bg-white/5"
        }`}
      />
      {/* side label, very subtle */}
      <div
        className={`absolute top-4 ${
          side === "left" ? "left-4" : "right-4"
        } text-[10px] uppercase tracking-wider text-white/50`}
      >
        {side === "left" ? "← A" : "D →"}
      </div>
      <div className="absolute inset-x-0 bottom-0 p-6 text-left">
        <div className="text-3xl sm:text-4xl font-semibold text-white drop-shadow-lg">
          {city.name}
        </div>
        <div className="text-sm text-white/70 mt-1">{city.country}</div>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs tabular-nums text-white/85">
          <Stat label="temp" value={`${city.stats.temp_c.toFixed(1)}°C`} />
          <Stat label="humid" value={`${city.stats.humidity_pct.toFixed(0)}%`} />
          <Stat label="rain" value={`${city.stats.precip_mm.toFixed(2)} mm/h`} />
          <Stat label="sun" value={`${city.stats.sun_pct.toFixed(0)}%`} />
        </div>
      </div>
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <span className="text-white/50">{label}</span>{" "}
      <span className="font-medium">{value}</span>
    </span>
  );
}

function BootLoader({ progress }: { progress: number }) {
  return (
    <div className="col-span-full flex flex-col items-center justify-center px-8 py-24 gap-6">
      <div className="text-center max-w-md">
        <div className="text-lg font-medium text-zinc-200">
          Waking up the climate server…
        </div>
        <div className="text-sm text-zinc-500 mt-2">
          The backend sleeps after 15 min of idle on the free tier. First
          visit takes <span className="text-zinc-300">30–60s</span>; after
          that, every pick is instant.
        </div>
      </div>
      <div className="w-full max-w-md space-y-2">
        <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-200"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-xs text-zinc-500 tabular-nums text-right">
          {progress.toFixed(0)}%
        </div>
      </div>
    </div>
  );
}
