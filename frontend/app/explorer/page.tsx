"use client";

import { useEffect, useRef, useState } from "react";
import {
  getPcaOverview,
  projectText,
  type PCAOverview,
  type ProjectedPoint,
} from "@/lib/api";

export default function ExplorerPage() {
  const [overview, setOverview] = useState<PCAOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [projected, setProjected] = useState<ProjectedPoint | null>(null);
  const [projecting, setProjecting] = useState(false);
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getPcaOverview()
      .then(setOverview)
      .catch((e) => setError((e as Error).message));
  }, []);

  useEffect(() => {
    if (!overview || !plotRef.current) return;

    let cancelled = false;
    (async () => {
      const Plotly = (await import("plotly.js-dist-min")).default;
      if (cancelled || !plotRef.current) return;

      const cityTrace = {
        x: overview.cities.map((c) => c.pc1),
        y: overview.cities.map((c) => c.pc2),
        text: overview.cities.map((c) => `${c.name}, ${c.country}`),
        mode: "markers",
        type: "scatter",
        marker: { size: 9, color: "#a5b4fc", opacity: 0.85 },
        hoverinfo: "text",
        name: "Cities",
      };

      const traces: unknown[] = [cityTrace];
      if (projected) {
        traces.push({
          x: [projected.pc1],
          y: [projected.pc2],
          text: [
            `your query${projected.anchor_name ? ` (${projected.anchor_name})` : ""}`,
          ],
          mode: "markers",
          type: "scatter",
          marker: { size: 18, color: "#f43f5e", symbol: "star" },
          hoverinfo: "text",
          name: "You",
        });
      }

      Plotly.newPlot(
        plotRef.current,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        traces as any,
        {
          autosize: true,
          height: 520,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#e4e4e7" },
          xaxis: {
            title: {
              text: `${overview.pc1_label} · ${(overview.explained_variance[0] * 100).toFixed(0)}% var`,
            },
            gridcolor: "#27272a",
            zerolinecolor: "#3f3f46",
          },
          yaxis: {
            title: {
              text: `${overview.pc2_label} · ${(overview.explained_variance[1] * 100).toFixed(0)}% var`,
            },
            gridcolor: "#27272a",
            zerolinecolor: "#3f3f46",
          },
          margin: { t: 20, r: 20, b: 50, l: 50 },
          showlegend: false,
        },
        { displayModeBar: false, responsive: true }
      );
    })();

    return () => {
      cancelled = true;
    };
  }, [overview, projected]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setProjecting(true);
    try {
      const p = await projectText(text);
      setProjected(p);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setProjecting(false);
    }
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-900 bg-red-950/40 p-4 text-red-300 text-sm">
        {error}
      </div>
    );
  }

  if (!overview) {
    return <div className="text-zinc-500">Loading PCA…</div>;
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Climate map</h1>
        <p className="mt-2 text-zinc-400">
          102 cities projected onto 2D via PCA. Similar climates cluster together.
        </p>
      </header>

      <form onSubmit={submit} className="flex gap-3">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="flex-1 rounded-lg bg-zinc-900 border border-zinc-800 px-4 py-3 text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
          placeholder="Plot a query on the map, e.g. London but sunnier"
        />
        <button
          type="submit"
          disabled={projecting || !text.trim()}
          className="rounded-lg bg-zinc-100 text-zinc-950 px-5 py-3 font-medium disabled:opacity-40 hover:bg-white transition"
        >
          {projecting ? "..." : "Plot"}
        </button>
      </form>

      <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-2">
        <div ref={plotRef} style={{ width: "100%", height: 520 }} />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <LoadingsPanel
          title={`PC1 · ${overview.pc1_label}`}
          loadings={overview.pc1_top}
        />
        <LoadingsPanel
          title={`PC2 · ${overview.pc2_label}`}
          loadings={overview.pc2_top}
        />
      </div>
    </div>
  );
}

function LoadingsPanel({
  title,
  loadings,
}: {
  title: string;
  loadings: PCAOverview["pc1_top"];
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-4">
      <h3 className="text-sm font-medium text-zinc-300 mb-3">{title}</h3>
      <ul className="space-y-1.5 text-sm">
        {loadings.map((l, i) => (
          <li key={i} className="flex items-center justify-between tabular-nums">
            <span className="text-zinc-400">{l.label}</span>
            <span className={l.weight >= 0 ? "text-emerald-400" : "text-rose-400"}>
              {l.weight >= 0 ? "+" : ""}
              {l.weight.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
