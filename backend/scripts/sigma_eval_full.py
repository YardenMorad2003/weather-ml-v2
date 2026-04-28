"""Comprehensive sigma-space eval: classical vs contrastive, 1-vibe vs 2-vibe.

Four modes:
  1. anchored 1-vibe   — 20 anchors x 10 axes x 3 scopes
  2. anchored 2-vibe   — 20 anchors x 40 pairs x 3 scope modes
  3. free-form 1-vibe  — 10 axes x 3 scopes
  4. free-form 2-vibe  — 40 pairs x 3 scope modes

For each query we generate (a) the structured form to compute the sigma-space
expected_delta and run the classical ranker, (b) a natural-language phrasing
to feed the contrastive ranker. Both rankers' top-K are scored against the
same structured ground truth.

We also apply the conflict + saturation guards (which only the classical
recommender uses in production) and report results on the post-guard "clean"
subset — that's what users actually see.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.state import get_state  # noqa: E402
from backend.services.vibe_table import (  # noqa: E402
    apply_vibes, detect_vibe_conflicts,
)
from backend.services.contrastive import CityEncoder, TextProjection  # noqa: E402
from backend.vendor.weather_ml.cities import CITIES  # noqa: E402


FOCUS_WEIGHT = 6.0
BASE_WEIGHT = 1.0
EPS = 1e-9
SAT_RATIO_THRESHOLD = 0.20
TOP_K = 10

CITY_NAMES = [c["name"] for c in CITIES]
NAME_TO_IDX = {n.lower(): i for i, n in enumerate(CITY_NAMES)}
CACHE = REPO_ROOT / "backend" / "cache"

ALL_AXES = [
    "warmer", "colder",
    "drier", "wetter",
    "sunnier", "cloudier",
    "windier", "calmer",
    "more_humid", "less_muggy",
]
HARD_OPPS = {
    frozenset({"warmer", "colder"}),
    frozenset({"drier", "wetter"}),
    frozenset({"sunnier", "cloudier"}),
    frozenset({"windier", "calmer"}),
    frozenset({"more_humid", "less_muggy"}),
}

AXIS_PHRASE = {
    "warmer": "warmer", "colder": "colder",
    "drier": "drier", "wetter": "rainier",
    "sunnier": "sunnier", "cloudier": "cloudier",
    "windier": "windier", "calmer": "less windy",
    "more_humid": "more humid", "less_muggy": "less humid",
}
SCOPE_PHRASE = {
    "year_round": "",
    "winter": " in winter",
    "summer": " in summer",
}


def phrase(anchor_name: str | None, vibes: list[dict]) -> str:
    parts = [
        f"{AXIS_PHRASE[v['axis']]}{SCOPE_PHRASE[v['scope']]}"
        for v in vibes
    ]
    body = " and ".join(parts)
    if anchor_name:
        return f"{anchor_name} but {body}"
    return f"a place that is {body}"


def classical_topk(profiles_scaled, anchor_idx, vibes):
    user = (
        profiles_scaled[anchor_idx].copy()
        if anchor_idx is not None
        else profiles_scaled.mean(axis=0)
    )
    mod, touched = apply_vibes(user, vibes)
    weights = BASE_WEIGHT + (FOCUS_WEIGHT - BASE_WEIGHT) * touched
    diffs = profiles_scaled - mod[None, :]
    dists = np.sqrt((weights[None, :] * diffs * diffs).sum(axis=1))
    order = np.argsort(dists).tolist()
    if anchor_idx is not None:
        order = [i for i in order if i != anchor_idx]
    return order[:TOP_K], user, mod, touched


def contrastive_topk(query_emb_proj, city_embs, anchor_idx):
    sims = (query_emb_proj @ city_embs.T)[0]
    order = np.argsort(-sims).tolist()
    if anchor_idx is not None:
        order = [i for i in order if i != anchor_idx]
    return order[:TOP_K]


def cosine_score(top, user, mod, touched, profiles_scaled):
    exp_d = (mod - user) * touched
    obs_d = (profiles_scaled[top].mean(axis=0) - user) * touched
    en = float(np.linalg.norm(exp_d))
    on = float(np.linalg.norm(obs_d))
    if en == 0:
        return None, en, on
    return float(np.dot(exp_d, obs_d) / (en * on + EPS)), en, on


def gen_anchored_1vibe(anchor_idxs):
    out = []
    for ai in anchor_idxs:
        for ax in ALL_AXES:
            for sc in ["year_round", "winter", "summer"]:
                out.append((ai, [{"axis": ax, "scope": sc, "intensity": "noticeably"}]))
    return out


def gen_anchored_2vibe(anchor_idxs):
    pairs = [
        (a, b) for a, b in itertools.combinations(ALL_AXES, 2)
        if frozenset({a, b}) not in HARD_OPPS
    ]
    scope_modes = [("year_round", "year_round"),
                   ("winter", "winter"),
                   ("winter", "summer")]
    out = []
    for ai in anchor_idxs:
        for ax1, ax2 in pairs:
            for sc1, sc2 in scope_modes:
                out.append((ai, [
                    {"axis": ax1, "scope": sc1, "intensity": "noticeably"},
                    {"axis": ax2, "scope": sc2, "intensity": "noticeably"},
                ]))
    return out


def gen_freeform_1vibe():
    out = []
    for ax in ALL_AXES:
        for sc in ["year_round", "winter", "summer"]:
            out.append((None, [{"axis": ax, "scope": sc, "intensity": "noticeably"}]))
    return out


def gen_freeform_2vibe():
    pairs = [
        (a, b) for a, b in itertools.combinations(ALL_AXES, 2)
        if frozenset({a, b}) not in HARD_OPPS
    ]
    scope_modes = [("year_round", "year_round"),
                   ("winter", "winter"),
                   ("winter", "summer")]
    out = []
    for ax1, ax2 in pairs:
        for sc1, sc2 in scope_modes:
            out.append((None, [
                {"axis": ax1, "scope": sc1, "intensity": "noticeably"},
                {"axis": ax2, "scope": sc2, "intensity": "noticeably"},
            ]))
    return out


def run_mode(name, queries, profiles_scaled, st_model, text_proj, city_embs):
    rows = []
    # Pre-encode all texts in one batch for speed.
    texts = [phrase(CITY_NAMES[ai] if ai is not None else None, vibes)
             for ai, vibes in queries]
    raw = st_model.encode(texts, convert_to_numpy=True,
                          normalize_embeddings=False, batch_size=64,
                          show_progress_bar=False).astype(np.float32)
    with torch.no_grad():
        proj_all = F.normalize(text_proj(torch.from_numpy(raw)), dim=1).numpy()

    for (ai, vibes), q_proj in zip(queries, proj_all):
        # Classical
        cls_top, user, mod, touched = classical_topk(profiles_scaled, ai, vibes)
        cls_cos, en, on = cosine_score(cls_top, user, mod, touched, profiles_scaled)
        if cls_cos is None:
            continue
        cls_ratio = on / en if en > 0 else 0.0

        # Contrastive
        ctr_top = contrastive_topk(q_proj[None, :], city_embs, ai)
        ctr_cos, _, on_ctr = cosine_score(ctr_top, user, mod, touched, profiles_scaled)
        ctr_ratio = on_ctr / en if en > 0 else 0.0

        # Guards (only apply to classical's served results — contrastive doesn't
        # use these in production today, but we apply the same conflict guard
        # for fairness since the parser would catch it before either ranker).
        is_conflict = bool(detect_vibe_conflicts(vibes))
        is_saturation = (cls_ratio < SAT_RATIO_THRESHOLD and cls_cos < 0)

        rows.append({
            "anchor_idx": ai, "vibes": vibes,
            "cls_cos": cls_cos, "ctr_cos": ctr_cos,
            "cls_ratio": cls_ratio, "ctr_ratio": ctr_ratio,
            "is_conflict": is_conflict,
            "is_saturation": is_saturation,
        })
    return rows


def fmt_stats(rows, key):
    arr = np.array([r[key] for r in rows])
    return {
        "n": len(arr),
        "mean": arr.mean() if len(arr) else 0.0,
        "median": np.median(arr) if len(arr) else 0.0,
        "p25": np.quantile(arr, 0.25) if len(arr) else 0.0,
        "p75": np.quantile(arr, 0.75) if len(arr) else 0.0,
        "ge_0.7": (arr >= 0.7).mean() * 100 if len(arr) else 0.0,
        "lt_0": (arr < 0).mean() * 100 if len(arr) else 0.0,
    }


def print_mode_stats(name, rows):
    served = [r for r in rows if not (r["is_conflict"] or r["is_saturation"])]
    n_total = len(rows)
    n_conf = sum(1 for r in rows if r["is_conflict"])
    n_sat = sum(1 for r in rows if r["is_saturation"])
    n_clean = len(served)

    cls_all = fmt_stats(rows, "cls_cos")
    ctr_all = fmt_stats(rows, "ctr_cos")
    cls_clean = fmt_stats(served, "cls_cos")
    ctr_clean = fmt_stats(served, "ctr_cos")

    print()
    print("=" * 80)
    print(f"  MODE: {name}")
    print("=" * 80)
    print(f"  total queries: {n_total}   guarded: {n_conf} conflict + "
          f"{n_sat} saturation = {n_conf+n_sat}   clean: {n_clean}")
    print()
    print(f"  {'metric':<20} {'classical':>12} {'contrastive':>14}  delta")
    print(f"  {'-'*20} {'-'*12} {'-'*14}  {'-'*7}")
    print(f"  raw mean cosine     {cls_all['mean']:+12.3f} {ctr_all['mean']:+14.3f}  "
          f"{cls_all['mean']-ctr_all['mean']:+.3f}")
    print(f"  raw median          {cls_all['median']:+12.3f} {ctr_all['median']:+14.3f}  "
          f"{cls_all['median']-ctr_all['median']:+.3f}")
    print(f"  raw <0 rate         {cls_all['lt_0']:11.1f}% {ctr_all['lt_0']:13.1f}%")
    print()
    print(f"  --- on the {n_clean} clean (post-guard) queries ---")
    print(f"  mean cosine         {cls_clean['mean']:+12.3f} {ctr_clean['mean']:+14.3f}  "
          f"{cls_clean['mean']-ctr_clean['mean']:+.3f}")
    print(f"  median              {cls_clean['median']:+12.3f} {ctr_clean['median']:+14.3f}  "
          f"{cls_clean['median']-ctr_clean['median']:+.3f}")
    print(f"  P25 / P75           {cls_clean['p25']:+5.2f}/{cls_clean['p75']:+5.2f} "
          f"  {ctr_clean['p25']:+5.2f}/{ctr_clean['p75']:+5.2f}")
    print(f"  fraction >= 0.7     {cls_clean['ge_0.7']:11.1f}% {ctr_clean['ge_0.7']:13.1f}%")
    print(f"  fraction <  0       {cls_clean['lt_0']:11.1f}% {ctr_clean['lt_0']:13.1f}%")


def main():
    print("loading state...", flush=True)
    profiles_scaled = get_state()["profiles_scaled"]
    print(f"  {len(profiles_scaled)} cities, profile dim {profiles_scaled.shape[1]}")

    print("loading contrastive artifacts...", flush=True)
    text_proj = TextProjection()
    text_proj.load_state_dict(torch.load(CACHE / "text_proj_state.pt", map_location="cpu"))
    text_proj.eval()
    emb_data = np.load(CACHE / "city_embeddings_learned.npz", allow_pickle=True)
    city_embs = emb_data["embeddings"]
    assert list(emb_data["cities"]) == CITY_NAMES

    print("loading sentence-transformer...", flush=True)
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )

    rng = np.random.RandomState(0)
    pinned = ["Tokyo", "Seattle", "Phoenix", "Cairo", "Reykjavik",
              "Singapore", "London", "Sydney"]
    ai = [NAME_TO_IDX[n.lower()] for n in pinned if n.lower() in NAME_TO_IDX]
    rest = [i for i in range(len(profiles_scaled)) if i not in set(ai)]
    rng.shuffle(rest)
    anchor_idxs = (ai + rest)[:20]

    modes = [
        ("anchored 1-vibe",  gen_anchored_1vibe(anchor_idxs)),
        ("anchored 2-vibe",  gen_anchored_2vibe(anchor_idxs)),
        ("free-form 1-vibe", gen_freeform_1vibe()),
        ("free-form 2-vibe", gen_freeform_2vibe()),
    ]

    summary = []
    for name, queries in modes:
        print(f"\n>>> running mode '{name}' ({len(queries)} queries)...", flush=True)
        rows = run_mode(
            name, queries, profiles_scaled, st_model, text_proj, city_embs
        )
        print_mode_stats(name, rows)
        served = [r for r in rows if not (r["is_conflict"] or r["is_saturation"])]
        cls_clean = fmt_stats(served, "cls_cos")
        ctr_clean = fmt_stats(served, "ctr_cos")
        summary.append({
            "mode": name, "n_total": len(rows), "n_clean": len(served),
            "cls_mean": cls_clean["mean"], "ctr_mean": ctr_clean["mean"],
            "cls_lt0": cls_clean["lt_0"], "ctr_lt0": ctr_clean["lt_0"],
        })

    print()
    print("=" * 80)
    print("  HEADLINE TABLE (post-guard, mean cosine)")
    print("=" * 80)
    print(f"  {'mode':<22} {'n':>5} {'classical':>12} {'contrastive':>13} {'delta':>9}")
    for s in summary:
        delta = s["cls_mean"] - s["ctr_mean"]
        print(f"  {s['mode']:<22} {s['n_clean']:>5} "
              f"{s['cls_mean']:+12.3f} {s['ctr_mean']:+13.3f} "
              f"{delta:+9.3f}")
    print()
    print(f"  {'mode':<22} {'n':>5} {'cls <0%':>12} {'ctr <0%':>13}")
    for s in summary:
        print(f"  {s['mode']:<22} {s['n_clean']:>5} "
              f"{s['cls_lt0']:11.1f}% {s['ctr_lt0']:12.1f}%")


if __name__ == "__main__":
    main()
