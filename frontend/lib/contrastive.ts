// Browser-side ranker for the Stage 1 contrastive model.
//
// Runtime path on a query:
//   1. MiniLM (frozen, quantized ONNX, ~13 MB) -> 384-d mean-pooled vector
//   2. Trained projection head (32 x 384) + bias -> 32-d
//   3. L2 normalize
//   4. Cosine vs precomputed L2-normed (230, 32) city embeddings -> top-K
//
// The 384-d MiniLM output is intentionally NOT L2-normalized before the
// projection: training used `normalize_embeddings=False`
// (see backend/scripts/precompute_embeddings.py). We must match that.
//
// Both the manifest fetch and the MiniLM pipeline are loaded lazily on the
// first call. preloadContrastive() lets a page warm them in the background
// from useEffect so the first query feels instant.

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

type Manifest = {
  cities: string[];
  embeddings: number[][];
  projection: { weight: number[][]; bias: number[] };
  scaler: { mean: number[]; scale: number[] };
  embed_dim: number;
  text_dim: number;
  city_input_dim: number;
};

type Encoder = (text: string) => Promise<Float32Array>;

type Loaded = {
  cities: string[];
  embedDim: number;
  textDim: number;
  cityEmb: Float32Array;
  projWeight: Float32Array;
  projBias: Float32Array;
  encode: Encoder;
};

let loadingPromise: Promise<Loaded> | null = null;

async function fetchManifest(): Promise<Manifest> {
  const r = await fetch(`${API_BASE}/contrastive/manifest`);
  if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
  return (await r.json()) as Manifest;
}

async function loadEncoder(): Promise<Encoder> {
  const { pipeline } = await import("@huggingface/transformers");
  const extractor = await pipeline("feature-extraction", MODEL_ID, {
    dtype: "q8",
  });
  return async (text) => {
    const out = await extractor(text, { pooling: "mean", normalize: false });
    return out.data as Float32Array;
  };
}

function flattenManifest(m: Manifest): Omit<Loaded, "encode"> {
  if (m.embeddings.length !== m.cities.length) {
    throw new Error("manifest: cities/embeddings length mismatch");
  }
  if (m.projection.weight.length !== m.embed_dim) {
    throw new Error("manifest: projection.weight rows != embed_dim");
  }
  if (m.projection.weight[0].length !== m.text_dim) {
    throw new Error("manifest: projection.weight cols != text_dim");
  }

  const cityEmb = new Float32Array(m.cities.length * m.embed_dim);
  for (let i = 0; i < m.cities.length; i++) {
    const row = m.embeddings[i];
    const off = i * m.embed_dim;
    for (let j = 0; j < m.embed_dim; j++) cityEmb[off + j] = row[j];
  }

  const projWeight = new Float32Array(m.embed_dim * m.text_dim);
  for (let i = 0; i < m.embed_dim; i++) {
    const row = m.projection.weight[i];
    const off = i * m.text_dim;
    for (let j = 0; j < m.text_dim; j++) projWeight[off + j] = row[j];
  }

  const projBias = Float32Array.from(m.projection.bias);

  return {
    cities: m.cities,
    embedDim: m.embed_dim,
    textDim: m.text_dim,
    cityEmb,
    projWeight,
    projBias,
  };
}

function load(): Promise<Loaded> {
  if (!loadingPromise) {
    loadingPromise = (async () => {
      const [manifest, encode] = await Promise.all([
        fetchManifest(),
        loadEncoder(),
      ]);
      return { ...flattenManifest(manifest), encode };
    })();
  }
  return loadingPromise;
}

/** Kick off model + manifest download in the background. Safe to call from
 *  useEffect on page mount; subsequent calls are no-ops. */
export function preloadContrastive(): void {
  void load();
}

export type ContrastiveResult = {
  city: string;
  similarity: number;
  rank: number;
};

export async function rankContrastive(
  query: string,
  topK = 10
): Promise<ContrastiveResult[]> {
  const { cities, embedDim, textDim, cityEmb, projWeight, projBias, encode } =
    await load();

  const text384 = await encode(query);
  if (text384.length !== textDim) {
    throw new Error(
      `encoder returned ${text384.length}-d, expected ${textDim}-d`
    );
  }

  // proj = projWeight (embedDim x textDim) @ text384 + projBias
  const proj = new Float32Array(embedDim);
  for (let i = 0; i < embedDim; i++) {
    let s = projBias[i];
    const off = i * textDim;
    for (let j = 0; j < textDim; j++) s += projWeight[off + j] * text384[j];
    proj[i] = s;
  }

  let norm = 0;
  for (let i = 0; i < embedDim; i++) norm += proj[i] * proj[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < embedDim; i++) proj[i] /= norm;

  // City embeddings are already L2-normed in the manifest; cosine = dot.
  const N = cities.length;
  const sims = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let s = 0;
    const off = n * embedDim;
    for (let d = 0; d < embedDim; d++) s += cityEmb[off + d] * proj[d];
    sims[n] = s;
  }

  const idxs = Array.from({ length: N }, (_, i) => i);
  idxs.sort((a, b) => sims[b] - sims[a]);
  const k = Math.min(topK, N);
  const out: ContrastiveResult[] = new Array(k);
  for (let r = 0; r < k; r++) {
    out[r] = { city: cities[idxs[r]], similarity: sims[idxs[r]], rank: r };
  }
  return out;
}
