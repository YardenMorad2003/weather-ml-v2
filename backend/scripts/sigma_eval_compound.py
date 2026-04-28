"""Compound (2-vibe) sigma-space eval for the classical recommender.

Same metric as sigma_eval.py but with two vibes per query. Filters out
conflicting axis pairs (warmer+colder, drier+wetter, etc) since those
queries are nonsensical. Enumerates all compatible pairs across three
scope modes:
  - both year_round
  - both winter
  - mixed (vibe 1 winter, vibe 2 summer)

Aggregates by axis-pair, scope-mode, and saturation guard hit-rate.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.state import get_state  # noqa: E402
from backend.services.vibe_table import apply_vibes  # noqa: E402
from backend.vendor.weather_ml.cities import CITIES  # noqa: E402


FOCUS_WEIGHT = 6.0
BASE_WEIGHT = 1.0
EPS = 1e-9
SAT_RATIO_THRESHOLD = 0.20

CITY_NAMES = [c["name"] for c in CITIES]
NAME_TO_IDX = {n.lower(): i for i, n in enumerate(CITY_NAMES)}

ALL_AXES = [
    "warmer", "colder",
    "drier", "wetter",
    "sunnier", "cloudier",
    "windier", "calmer",
    "more_humid", "less_muggy",
]

# Pairs that contradict each other — skip these.
CONFLICTING_PAIRS = {
    frozenset({"warmer", "colder"}),
    frozenset({"drier", "wetter"}),
    frozenset({"sunnier", "cloudier"}),
    frozenset({"windier", "calmer"}),
    frozenset({"more_humid", "less_muggy"}),
    # less_muggy ⊂ drier in feature terms (both push humidity down) — keep
    # them combinable; they just emphasize different features.
}


def evaluate(anchor_idx, vibes, profiles_scaled, top_k=10):
    user = profiles_scaled[anchor_idx].copy()
    mod, touched = apply_vibes(user, vibes)
    weights = BASE_WEIGHT + (FOCUS_WEIGHT - BASE_WEIGHT) * touched
    diffs = profiles_scaled - mod[None, :]
    dists = np.sqrt((weights[None, :] * diffs * diffs).sum(axis=1))
    order = np.argsort(dists).tolist()
    top = [i for i in order if i != anchor_idx][:top_k]

    exp_d = (mod - user) * touched
    obs_d = (profiles_scaled[top].mean(axis=0) - user) * touched
    exp_n = float(np.linalg.norm(exp_d))
    obs_n = float(np.linalg.norm(obs_d))
    cos = float(np.dot(exp_d, obs_d) / (exp_n * obs_n + EPS)) if exp_n > 0 else 0.0
    return cos, exp_n, obs_n, top


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-anchors", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    print("loading state...", flush=True)
    profiles_scaled = get_state()["profiles_scaled"]
    n = len(profiles_scaled)
    print(f"  {n} cities, dim {profiles_scaled.shape[1]}")

    # Enumerate compatible axis pairs (unordered).
    pairs = []
    for a, b in itertools.combinations(ALL_AXES, 2):
        if frozenset({a, b}) in CONFLICTING_PAIRS:
            continue
        pairs.append((a, b))
    print(f"  {len(pairs)} compatible axis pairs (after filtering conflicts)")

    # Pinned anchors + random fill.
    pinned = ["Tokyo", "Seattle", "Phoenix", "Cairo", "Reykjavik",
              "Singapore", "London", "Sydney"]
    ai = [NAME_TO_IDX[p.lower()] for p in pinned if p.lower() in NAME_TO_IDX]
    rest = [i for i in range(n) if i not in set(ai)]
    rng.shuffle(rest)
    anchor_idxs = (ai + rest)[: args.n_anchors]

    # Scope modes for compound queries.
    scope_modes = [
        ("year_year",   "year_round", "year_round"),
        ("winter_winter", "winter",   "winter"),
        ("winter_summer", "winter",   "summer"),
    ]

    queries = []
    for ai_ in anchor_idxs:
        for ax1, ax2 in pairs:
            for mode_name, sc1, sc2 in scope_modes:
                queries.append((ai_, ax1, sc1, ax2, sc2, mode_name))

    print(f"running {len(queries)} compound (2-vibe) queries "
          f"({len(anchor_idxs)} anchors x {len(pairs)} pairs x "
          f"{len(scope_modes)} scope modes)\n")

    rows = []
    for ai_, ax1, sc1, ax2, sc2, mode in queries:
        vibes = [
            {"axis": ax1, "scope": sc1, "intensity": "noticeably"},
            {"axis": ax2, "scope": sc2, "intensity": "noticeably"},
        ]
        cos, exp_n, obs_n, top = evaluate(ai_, vibes, profiles_scaled, args.top_k)
        rows.append({
            "anchor": CITY_NAMES[ai_],
            "ax1": ax1, "sc1": sc1, "ax2": ax2, "sc2": sc2, "mode": mode,
            "pair": f"{ax1}+{ax2}",
            "cos": cos, "exp_n": exp_n, "obs_n": obs_n,
            "ratio": (obs_n / exp_n) if exp_n > 0 else 0.0,
            "top": [CITY_NAMES[i] for i in top[:5]],
        })

    cosines = np.array([r["cos"] for r in rows])
    ratios = np.array([r["ratio"] for r in rows])

    print("== overall ==")
    print(f"  mean cosine:        {cosines.mean():+.3f}")
    print(f"  median:             {np.median(cosines):+.3f}")
    print(f"  P25 / P75:          {np.quantile(cosines, 0.25):+.3f} / "
          f"{np.quantile(cosines, 0.75):+.3f}")
    print(f"  fraction >= 0.7:    {(cosines >= 0.7).mean()*100:5.1f}%")
    print(f"  fraction >= 0.9:    {(cosines >= 0.9).mean()*100:5.1f}%")
    print(f"  fraction <  0.3:    {(cosines <  0.3).mean()*100:5.1f}%")
    print(f"  fraction <  0:      {(cosines <  0  ).mean()*100:5.1f}% "
          f"(directional failure)")
    sat = ((ratios < SAT_RATIO_THRESHOLD) & (cosines < 0))
    print(f"  saturation guard:   would fire on {sat.sum()}/{len(rows)} "
          f"= {sat.mean()*100:.1f}%")
    eff_fail = ((cosines < 0) & ~sat).mean() * 100
    print(f"  user-facing fail:   {eff_fail:.1f}%   "
          "(directional fail not caught by guard)")
    print(f"  mean ratio (|obs|/|exp|): {ratios.mean():.2f}   "
          "(magnitude saturation indicator)")

    print()
    print("== by scope mode ==")
    print(f"  {'mode':<16} {'mean':>8} {'median':>8} {'<0%':>6} "
          f"{'guard%':>8} {'n':>5}")
    for mode_name, _, _ in scope_modes:
        sub = [r for r in rows if r["mode"] == mode_name]
        c = np.array([r["cos"] for r in sub])
        rt = np.array([r["ratio"] for r in sub])
        sat_sub = ((rt < SAT_RATIO_THRESHOLD) & (c < 0)).mean() * 100
        print(f"  {mode_name:<16} {c.mean():+8.3f} {np.median(c):+8.3f} "
              f"{(c<0).mean()*100:5.1f} {sat_sub:7.1f} {len(sub):>5}")

    print()
    print("== by axis pair (sorted by mean cosine, asc) ==")
    print(f"  {'pair':<28} {'mean':>8} {'median':>8} {'<0%':>6} {'n':>5}")
    pair_stats = []
    for pair in sorted(set(r["pair"] for r in rows)):
        sub = [r["cos"] for r in rows if r["pair"] == pair]
        a = np.array(sub)
        pair_stats.append((pair, a.mean(), np.median(a), (a<0).mean()*100, len(sub)))
    for pair, m, md, fl, ct in sorted(pair_stats, key=lambda x: x[1]):
        print(f"  {pair:<28} {m:+8.3f} {md:+8.3f} {fl:5.1f} {ct:>5}")

    print()
    print("== 10 worst ==")
    for r in sorted(rows, key=lambda r: r["cos"])[:10]:
        print(f"  cos={r['cos']:+.2f} ratio={r['ratio']:.2f}  "
              f"{r['anchor']:<14} + {r['ax1']}/{r['sc1']} + "
              f"{r['ax2']}/{r['sc2']}: {r['top']}")
    print()
    print("== 10 best ==")
    for r in sorted(rows, key=lambda r: -r["cos"])[:10]:
        print(f"  cos={r['cos']:+.2f} ratio={r['ratio']:.2f}  "
              f"{r['anchor']:<14} + {r['ax1']}/{r['sc1']} + "
              f"{r['ax2']}/{r['sc2']}: {r['top']}")


if __name__ == "__main__":
    main()
