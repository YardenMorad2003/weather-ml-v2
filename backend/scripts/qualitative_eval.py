"""Qualitative + categorical evaluation of the trained contrastive model.

Three things, all in one script so the demo deck has one source of truth:

  1. Probe queries — top-10 cities for a curated demo list (README examples
     plus a few diverse styles). Pure reality check; no metric, just stare
     at it and ask "does this look right?".

  2. Per-query-type recall@10 — partitions the 1k held-out test split into
     four styles (anchored, descriptive, multi-anchor, numerical) via simple
     regex and reports recall@10 per bucket. Surfaces failure modes the
     average hides.

  3. t-SNE visualization — embeds the trained 230x32 city vectors into 2D
     and saves a PNG colored by continent (derived from lat/lon, not country
     code, so we don't need a country->continent table). Demo slide #1.

Usage:

    python -m backend.scripts.qualitative_eval --out-dir backend/cache

Outputs:
    - stdout: probe results + recall-by-type table
    - backend/cache/contrastive_tsne.png
    - backend/cache/contrastive_tsne_coords.json   # (230, 2) for frontend
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.contrastive import (  # noqa: E402
    CityEncoder, TextProjection, EMBED_DIM,
)
from backend.vendor.weather_ml.cities import CITIES  # noqa: E402

CITY_NAMES = [c["name"] for c in CITIES]
CITY_NAME_TO_IDX = {n: i for i, n in enumerate(CITY_NAMES)}


# Curated probe set: README examples + a few extras spanning the query types
# the model was trained on. Order is deck-friendly, not random.
PROBE_QUERIES = [
    # README examples
    "Seattle but sunnier",
    "New York but warmer winters",
    "Tokyo but less humid summers",
    "like Tokyo",
    # Anchored variants
    "somewhere like Bangkok but with cooler nights",
    "like Phoenix but with milder winters",
    # Descriptive
    "warm coastal city with lots of sunshine",
    "tropical cities with monsoons",
    "cold dry continental winters",
    # Preference / list
    "I want hot dry summers and cold snowy winters",
    # Multi-anchor
    "between Barcelona and Miami",
    # Numerical
    "cities with winter highs above 50F and under 600mm of rain",
]


def lat_lon_to_continent(lat: float, lon: float) -> str:
    """Coarse continent bucket from coordinates. Good enough for color codes
    on a 230-point t-SNE plot; not authoritative geographically."""
    if -55 <= lat <= 15 and -85 <= lon <= -34:
        return "South America"
    if 15 < lat <= 72 and -170 <= lon <= -50:
        return "North America"
    if 35 <= lat <= 72 and -10 <= lon <= 60:
        return "Europe"
    if -35 <= lat <= 38 and -20 <= lon <= 52:
        return "Africa"
    if 0 <= lat <= 72 and 60 <= lon <= 180:
        return "Asia"
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return "Oceania"
    if lat >= 60 and -10 <= lon <= 60:
        return "Europe"  # Reykjavik, Murmansk
    return "Other"


def classify_query(q: str) -> str:
    """Coarse 4-way query type classifier via regex on the query text.

    Order matters — first match wins.
    """
    q_low = q.lower()

    # Numerical: any digit OR explicit unit phrasing
    if re.search(r"\d", q_low) or re.search(r"\b(mm|inches|degrees|fahrenheit|celsius)\b", q_low):
        return "numerical"

    # Multi-anchor: between X and Y
    if re.search(r"\bbetween\b.*\band\b", q_low):
        return "multi_anchor"

    # Anchored: like X / similar to X / X but / comparable to X
    if re.search(r"\b(like|similar to|comparable to|compared to|than)\b", q_low):
        return "anchored"
    if re.search(r"\bbut\b", q_low):
        return "anchored"

    return "descriptive"


def load_artifacts(cache_dir: Path):
    import torch
    text_proj = TextProjection()
    text_proj.load_state_dict(torch.load(cache_dir / "text_proj_state.pt", map_location="cpu"))
    text_proj.eval()

    city_enc = CityEncoder()
    city_enc.load_state_dict(torch.load(cache_dir / "city_encoder_state.pt", map_location="cpu"))
    city_enc.eval()

    emb_data = np.load(cache_dir / "city_embeddings_learned.npz", allow_pickle=True)
    city_embs = emb_data["embeddings"]                       # (230, 32) L2-normed
    cities_in_cache = list(emb_data["cities"])
    assert cities_in_cache == CITY_NAMES, "city order mismatch between cache and CITIES"

    return text_proj, city_enc, city_embs


def encode_queries(queries: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return model.encode(queries, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)


def project_and_score(text_proj, raw_text_embs: np.ndarray, city_embs: np.ndarray) -> np.ndarray:
    """raw_text_embs (N, 384) -> projected (N, 32) -> cosine vs (230, 32) -> (N, 230) sims."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        x = torch.from_numpy(raw_text_embs)
        proj = F.normalize(text_proj(x), dim=1).numpy()
    return proj @ city_embs.T  # both L2-normed -> cosine


def run_probes(text_proj, city_embs):
    print("\n" + "=" * 78)
    print("PROBE QUERIES — top-10 cities (cosine in 32-d learned space)")
    print("=" * 78)
    raw = encode_queries(PROBE_QUERIES)
    sims = project_and_score(text_proj, raw, city_embs)
    for i, q in enumerate(PROBE_QUERIES):
        order = np.argsort(-sims[i])[:10]
        cells = [f"{CITY_NAMES[j]} ({sims[i][j]:.2f})" for j in order]
        print(f"\n[{classify_query(q):>11}] {q}")
        print("  " + ", ".join(cells))


def run_per_type_recall(text_proj, city_embs, test_split_path: Path, k: int = 10):
    print("\n" + "=" * 78)
    print(f"RECALL@{k} BY QUERY TYPE  (held-out 1k split)")
    print("=" * 78)
    rows = [json.loads(line) for line in test_split_path.open(encoding="utf-8") if line.strip()]
    print(f"  loaded {len(rows)} test queries")

    queries = [r["query"] for r in rows]
    pos_idxs_per_query = [
        [CITY_NAME_TO_IDX[n] for n in r["positives"] if n in CITY_NAME_TO_IDX]
        for r in rows
    ]
    types = [classify_query(q) for q in queries]

    raw = encode_queries(queries)
    sims = project_and_score(text_proj, raw, city_embs)

    by_type: dict[str, list[int]] = {}  # type -> list of 0/1 hit
    for i, t in enumerate(types):
        topk = np.argsort(-sims[i])[:k].tolist()
        hit = int(any(p in topk for p in pos_idxs_per_query[i]))
        by_type.setdefault(t, []).append(hit)

    print(f"\n  {'type':<14} {'count':>6}  {f'recall@{k}':>10}")
    print(f"  {'-'*14} {'-'*6}  {'-'*10}")
    for t in ["anchored", "descriptive", "multi_anchor", "numerical"]:
        hits = by_type.get(t, [])
        if not hits:
            continue
        r = sum(hits) / len(hits)
        print(f"  {t:<14} {len(hits):>6}  {r:>10.3f}")
    overall = [h for hits in by_type.values() for h in hits]
    print(f"  {'-'*14} {'-'*6}  {'-'*10}")
    print(f"  {'OVERALL':<14} {len(overall):>6}  {sum(overall)/len(overall):>10.3f}")


def run_tsne(city_embs: np.ndarray, out_dir: Path, seed: int = 42):
    print("\n" + "=" * 78)
    print("t-SNE OF 230 CITY EMBEDDINGS (learned 32-d -> 2-d)")
    print("=" * 78)
    from sklearn.manifold import TSNE
    # perplexity 30 is typical for n~200; we have 230, fine.
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca",
                metric="cosine", learning_rate="auto")
    coords = tsne.fit_transform(city_embs)
    print(f"  coords shape: {coords.shape}")

    continents = [lat_lon_to_continent(c["lat"], c["lon"]) for c in CITIES]
    coord_payload = [
        {"city": CITY_NAMES[i], "x": float(coords[i, 0]), "y": float(coords[i, 1]),
         "continent": continents[i], "country": CITIES[i]["country"]}
        for i in range(len(CITY_NAMES))
    ]
    json_path = out_dir / "contrastive_tsne_coords.json"
    json_path.write_text(json.dumps(coord_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping PNG. Coords saved for frontend.")
        return

    palette = {
        "North America": "#1f77b4",
        "South America": "#ff7f0e",
        "Europe":        "#2ca02c",
        "Africa":        "#d62728",
        "Asia":          "#9467bd",
        "Oceania":       "#8c564b",
        "Other":         "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    for cont, color in palette.items():
        mask = np.array([c == cont for c in continents])
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22, c=color, label=cont,
                   alpha=0.85, edgecolor="white", linewidth=0.4)
    # Light city labels for landmark-y picks so the slide is readable
    landmark_set = {
        "Tokyo", "Seattle", "New York", "Reykjavik", "Cairo", "Singapore",
        "Phoenix", "Buenos Aires", "London", "Sydney", "Mumbai", "Anchorage",
        "Mexico City", "Cape Town", "Dubai",
    }
    for i, name in enumerate(CITY_NAMES):
        if name in landmark_set:
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.85, xytext=(3, 2), textcoords="offset points")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of contrastive city embeddings (230 cities, 32-d -> 2-d)")
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    png_path = out_dir / "contrastive_tsne.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {png_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=Path("backend/cache"))
    p.add_argument("--test-split", type=Path,
                   default=Path("backend/cache/contrastive_test_split.jsonl"))
    p.add_argument("--skip-tsne", action="store_true")
    p.add_argument("--skip-probes", action="store_true")
    p.add_argument("--skip-per-type", action="store_true")
    args = p.parse_args()

    text_proj, city_enc, city_embs = load_artifacts(args.out_dir)

    if not args.skip_probes:
        run_probes(text_proj, city_embs)
    if not args.skip_per_type:
        run_per_type_recall(text_proj, city_embs, args.test_split)
    if not args.skip_tsne:
        run_tsne(city_embs, args.out_dir)


if __name__ == "__main__":
    main()
