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

export type RecommendResponse = {
  parsed: ParsedQuery;
  anchor: Anchor | null;
  results: CityResult[];
};

export type CityPoint = {
  name: string;
  country: string;
  lat: number;
  lon: number;
  pc1: number;
  pc2: number;
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
  pc1_label: string;
  pc2_label: string;
  explained_variance: number[];
};

export type ProjectedPoint = {
  pc1: number;
  pc2: number;
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
