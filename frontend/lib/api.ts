const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Vibe = {
  axis: string;
  scope: string;
  intensity: string;
};

export type ParsedQuery = {
  anchor_city: string | null;
  vibes: Vibe[];
};

export type Anchor = {
  name: string;
  country: string;
  lat: number;
  lon: number;
  source: string;
};

export type Reason = {
  label: string;
  detail: string;
  matched: boolean;
};

export type CityResult = {
  city: string;
  country: string;
  lat: number;
  lon: number;
  similarity: number;
  reasons: Reason[];
};

export type AnchorError = {
  input: string;
  suggestions: string[];
};

export type RecommendResponse = {
  parsed: ParsedQuery;
  anchor: Anchor | null;
  results: CityResult[];
  anchor_error: AnchorError | null;
};

export type CityPoint = {
  name: string;
  country: string;
  lat: number;
  lon: number;
  pc1: number;
  pc2: number;
  pc3: number;
};

export type Loading = {
  label: string;
  feature: string;
  month: string;
  weight: number;
};

export type PCAOverview = {
  cities: CityPoint[];
  pc1_top: Loading[];
  pc2_top: Loading[];
  pc3_top: Loading[];
  pc1_label: string;
  pc2_label: string;
  pc3_label: string;
  pc1_explanation: string;
  pc2_explanation: string;
  pc3_explanation: string;
  explained_variance: number[];
};

export type ProjectedPoint = {
  pc1: number;
  pc2: number;
  pc3: number;
  anchor_name: string | null;
  parsed: ParsedQuery;
};

export async function recommendText(text: string, topK = 10): Promise<RecommendResponse> {
  const r = await fetch(`${BASE}/recommend/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, top_k: topK }),
  });
  if (!r.ok) throw new Error(`recommend failed: ${r.status}`);
  return r.json();
}

export async function getPcaOverview(): Promise<PCAOverview> {
  const r = await fetch(`${BASE}/pca`);
  if (!r.ok) throw new Error(`pca failed: ${r.status}`);
  return r.json();
}

export type AnnualStats = {
  temp_c: number;
  humidity_pct: number;
  precip_mm: number;
  wind_kmh: number;
  clear_sky_frac: number;
};

export type CityDetail = {
  name: string;
  country: string;
  lat: number;
  lon: number;
  annual: AnnualStats;
};

export async function getCity(name: string): Promise<CityDetail> {
  const r = await fetch(`${BASE}/cities/${encodeURIComponent(name)}`);
  if (!r.ok) throw new Error(`city lookup failed: ${r.status}`);
  return r.json();
}

export type WikiSummary = {
  thumbnail?: string;
  extract?: string;
};

export async function getWikiSummary(name: string): Promise<WikiSummary> {
  try {
    const r = await fetch(
      `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(name)}`
    );
    if (!r.ok) return {};
    const j = await r.json();
    return {
      thumbnail: j.thumbnail?.source,
      extract: j.extract,
    };
  } catch {
    return {};
  }
}

export async function projectText(text: string): Promise<ProjectedPoint> {
  const r = await fetch(`${BASE}/pca/project`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`project failed: ${r.status}`);
  return r.json();
}

export type TournamentHistoryItem = {
  shown: [string, string];
  picked: string;
};

export type CityStats = {
  temp_c: number;
  humidity_pct: number;
  precip_mm: number;
  sun_pct: number;
};

export type PairCity = {
  name: string;
  country: string;
  lat: number;
  lon: number;
  image_url: string;
  thumb_url: string;
  stats: CityStats;
};

export type PairResponse = {
  round: number;
  total_rounds: number;
  pair: [PairCity, PairCity];
};

export type FinalResponse = {
  rounds_completed: number;
  picked: string[];
  results: CityResult[];
};

/** Retry POST for cold-start resilience. Render free tier returns 502/503
 * while booting; browser fetches on dead containers throw TypeError. We
 * retry those for ~90s total; other errors (4xx, bad JSON) fail fast. */
async function postWithRetry(
  url: string,
  body: unknown,
  { maxAttempts = 6, onAttempt }: { maxAttempts?: number; onAttempt?: (n: number) => void } = {}
): Promise<Response> {
  const delays = [1500, 3000, 5000, 10000, 15000]; // cumulative ~35s; max sleep before 6th = +30s
  let lastErr: unknown = null;
  for (let i = 0; i < maxAttempts; i++) {
    onAttempt?.(i + 1);
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (r.ok) return r;
      // 5xx or gateway errors during boot: retry. 4xx: user/client bug, give up.
      if (r.status >= 500 && r.status < 600) {
        lastErr = new Error(`${url} ${r.status}`);
      } else {
        throw new Error(`${url} ${r.status}`);
      }
    } catch (e) {
      // TypeError "Failed to fetch" = network-level error, retry
      lastErr = e;
    }
    if (i < maxAttempts - 1) {
      await new Promise((res) => setTimeout(res, delays[Math.min(i, delays.length - 1)]));
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

export async function getTournamentPair(
  history: TournamentHistoryItem[],
  seed?: number,
  opts?: { onAttempt?: (n: number) => void }
): Promise<PairResponse> {
  const r = await postWithRetry(
    `${BASE}/tournament/pair`,
    { history, seed },
    { onAttempt: opts?.onAttempt }
  );
  return r.json();
}

export async function getTournamentFinal(
  history: TournamentHistoryItem[]
): Promise<FinalResponse> {
  const r = await postWithRetry(`${BASE}/tournament/final`, { history });
  return r.json();
}
