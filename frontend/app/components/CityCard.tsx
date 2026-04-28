"use client";

import { useEffect, useRef, useState } from "react";
import {
  getCity,
  getWikiSummary,
  type Anchor,
  type CityDetail,
  type CityResult,
  type WikiSummary,
} from "@/lib/api";

type Props = {
  rank: number;
  result: CityResult;
  anchor?: Anchor | null;
  expanded: boolean;
  onToggle: () => void;
};

export function CityCard({ rank, result, anchor, expanded, onToggle }: Props) {
  const [detail, setDetail] = useState<CityDetail | null>(null);
  const [wiki, setWiki] = useState<WikiSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!expanded || detail) return;
    setLoading(true);
    Promise.all([getCity(result.city), getWikiSummary(result.city)])
      .then(([d, w]) => {
        setDetail(d);
        setWiki(w);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [expanded, detail, result.city]);

  useEffect(() => {
    if (!expanded || !detail || !mapRef.current) return;

    let cancelled = false;
    (async () => {
      const Plotly = (await import("plotly.js-dist-min")).default;
      if (cancelled || !mapRef.current) return;

      // Result marker (rose). Plot the anchor as a second marker (sky blue)
      // when one is provided AND it's a different city — gives a visual sense
      // of how far the recommendation is from where the user started.
      const showAnchor =
        anchor &&
        anchor.name.toLowerCase() !== detail.name.toLowerCase();
      const traces: object[] = [
        {
          type: "scattergeo",
          mode: "markers",
          lat: [detail.lat],
          lon: [detail.lon],
          marker: {
            size: 10,
            color: "#f43f5e",
            line: { color: "#fff", width: 1 },
          },
          name: detail.name,
          text: [detail.name],
          hoverinfo: "text",
        },
      ];
      if (showAnchor) {
        traces.push({
          type: "scattergeo",
          mode: "markers",
          lat: [anchor.lat],
          lon: [anchor.lon],
          marker: {
            size: 9,
            color: "#38bdf8",
            symbol: "diamond",
            line: { color: "#fff", width: 1 },
          },
          name: `${anchor.name} (anchor)`,
          text: [`${anchor.name} (anchor)`],
          hoverinfo: "text",
        });
      }

      Plotly.newPlot(
        mapRef.current,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        traces as any,
        {
          height: 220,
          margin: { t: 0, r: 0, b: 0, l: 0 },
          paper_bgcolor: "rgba(0,0,0,0)",
          showlegend: false,
          geo: {
            projection: { type: "natural earth" },
            showland: true,
            landcolor: "#1f1f22",
            showcountries: true,
            countrycolor: "#3f3f46",
            showocean: true,
            oceancolor: "#09090b",
            coastlinecolor: "#3f3f46",
            showframe: false,
            bgcolor: "rgba(0,0,0,0)",
          },
        },
        { displayModeBar: false, responsive: true }
      );
    })();

    return () => {
      cancelled = true;
    };
  }, [expanded, detail, anchor]);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 overflow-hidden transition">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-zinc-900/70 transition text-left"
      >
        <div className="flex items-center gap-4">
          <span className="text-zinc-500 tabular-nums w-6">{rank}</span>
          <div>
            <div className="font-medium">{result.city}</div>
            <div className="text-xs text-zinc-500">{result.country}</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-zinc-400 tabular-nums">
            {(result.similarity * 100).toFixed(1)}%
          </span>
          <span
            className={`text-zinc-500 transition-transform ${
              expanded ? "rotate-180" : ""
            }`}
          >
            ▾
          </span>
        </div>
      </button>

      {expanded && (
        <div className="border-t border-zinc-800 p-4 space-y-4">
          {loading && !detail && (
            <div className="text-sm text-zinc-500">Loading…</div>
          )}

          {detail && (
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3">
                {wiki?.thumbnail ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={wiki.thumbnail}
                    alt={detail.name}
                    className="w-full h-40 object-cover rounded-lg border border-zinc-800"
                  />
                ) : (
                  <div className="w-full h-40 rounded-lg border border-zinc-800 bg-zinc-900 flex items-center justify-center text-xs text-zinc-600">
                    no photo
                  </div>
                )}
                {wiki?.extract && (
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {wiki.extract}
                  </p>
                )}
              </div>

              <div className="space-y-3">
                <div ref={mapRef} style={{ width: "100%", height: 220 }} />
                {anchor &&
                  detail &&
                  anchor.name.toLowerCase() !== detail.name.toLowerCase() && (
                    <div className="flex items-center gap-4 text-[10px] text-zinc-500">
                      <span className="flex items-center gap-1.5">
                        <span
                          className="inline-block w-2 h-2 rounded-full"
                          style={{ backgroundColor: "#f43f5e" }}
                        />
                        {detail.name}
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span
                          className="inline-block w-2 h-2 rotate-45"
                          style={{ backgroundColor: "#38bdf8" }}
                        />
                        {anchor.name} (your anchor)
                      </span>
                    </div>
                  )}
                {result.reasons.length > 0 && (
                  <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
                    <div className="flex items-center gap-1.5 mb-2">
                      <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                        Why this matches
                      </span>
                      <button
                        type="button"
                        onClick={() => setHelpOpen((o) => !o)}
                        aria-expanded={helpOpen}
                        aria-label="Explain these markers"
                        className="w-4 h-4 rounded-full border border-zinc-700 text-zinc-500 hover:text-zinc-200 hover:border-zinc-500 text-[10px] leading-none flex items-center justify-center transition"
                      >
                        ?
                      </button>
                    </div>
                    {helpOpen && (
                      <div className="mb-3 rounded border border-zinc-800 bg-zinc-900/60 p-2.5 text-[11px] text-zinc-400 leading-relaxed space-y-2">
                        <div className="text-zinc-300">
                          Each row compares this city to your starting city on
                          one weather dimension.
                        </div>
                        <div>
                          <span className="text-emerald-400">●</span>{" "}
                          <span className="text-zinc-300">green — fits.</span>
                          <div className="pl-3 mt-0.5 text-zinc-500">
                            If you asked to change this dimension, the city
                            moved the way you asked. Otherwise, the city stays
                            close to your starting city here.
                          </div>
                        </div>
                        <div>
                          <span className="text-zinc-600">●</span>{" "}
                          <span className="text-zinc-300">
                            gray — trade-off.
                          </span>
                          <div className="pl-3 mt-0.5 text-zinc-500">
                            The city is noticeably different from your starting
                            city here — enough to count as a real trade-off,
                            not just normal variation between cities. The +/−
                            shows which way it differs — not whether
                            that&apos;s good or bad.
                          </div>
                        </div>
                      </div>
                    )}
                    <ul className="space-y-1.5 text-xs">
                      {result.reasons.map((r, i) => (
                        <li key={i} className="flex items-baseline gap-2">
                          <span
                            className={
                              r.matched ? "text-emerald-400" : "text-zinc-600"
                            }
                            aria-hidden
                          >
                            ●
                          </span>
                          <span>
                            <span className="text-zinc-300 font-medium">
                              {r.label}:
                            </span>{" "}
                            <span className="text-zinc-400 tabular-nums">
                              {r.detail}
                            </span>
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="md:col-span-2 grid grid-cols-2 sm:grid-cols-5 gap-3">
                <Stat label="Avg temp" value={`${detail.annual.temp_c.toFixed(1)}°C`} />
                <Stat label="Humidity" value={`${detail.annual.humidity_pct.toFixed(0)}%`} />
                <Stat label="Precip" value={`${detail.annual.precip_mm.toFixed(2)} mm/h`} />
                <Stat label="Wind" value={`${detail.annual.wind_kmh.toFixed(1)} km/h`} />
                <Stat
                  label="Clear sky"
                  value={`${(detail.annual.clear_sky_frac * 100).toFixed(0)}%`}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="text-sm font-medium tabular-nums">{value}</div>
    </div>
  );
}
