"""Climatology-grounded synthetic metric for the classical recommender.

Implements EVAL_PLAN.md approach #2. For each (anchor, axis, scope, intensity)
tuple, computes the cosine similarity between:
  - expected_delta: the sigma-space shift apply_vibes() produces on touched dims
  - observed_delta: (mean of top-K profiles in sigma-space) - anchor profile
                    in sigma-space, restricted to the same touched dims

Bypasses nl_parser and city_resolver so the metric measures ranker fidelity
to the sigma-space delta, not parser/resolver noise.

Usage:
    python -m backend.scripts.sigma_eval --n-anchors 20 --top-k 10
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

CITY_NAMES = [c["name"] for c in CITIES]
NAME_TO_IDX = {n.lower(): i for i, n in enumerate(CITY_NAMES)}


def evaluate(
    anchor_idx: int,
    vibe: dict,
    profiles_scaled: np.ndarray,
    top_k: int,
) -> dict:
    user_scaled = profiles_scaled[anchor_idx].copy()
    user_modified, touched = apply_vibes(user_scaled, [vibe])
    expected_delta = (user_modified - user_scaled) * touched

    weights = BASE_WEIGHT + (FOCUS_WEIGHT - BASE_WEIGHT) * touched
    diffs = profiles_scaled - user_modified[None, :]
    dists = np.sqrt((weights[None, :] * diffs * diffs).sum(axis=1))

    order = np.argsort(dists).tolist()
    top = [i for i in order if i != anchor_idx][:top_k]

    observed_centroid = profiles_scaled[top].mean(axis=0)
    observed_delta = (observed_centroid - user_scaled) * touched

    exp_norm = float(np.linalg.norm(expected_delta))
    obs_norm = float(np.linalg.norm(observed_delta))
    cos = float(np.dot(expected_delta, observed_delta) / (exp_norm * obs_norm + EPS))

    return {
        "cosine": cos,
        "expected_norm": exp_norm,
        "observed_norm": obs_norm,
        "top": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-anchors", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axes", nargs="+", default=None)
    parser.add_argument("--show-examples", type=int, default=8)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    print("loading state (sigma-space profiles)...", flush=True)
    st = get_state()
    profiles_scaled = st["profiles_scaled"]
    n_cities = len(profiles_scaled)
    print(f"  loaded {n_cities} cities, profile dim {profiles_scaled.shape[1]}")

    test_axes = args.axes or [
        "warmer", "colder",
        "drier", "wetter",
        "sunnier", "cloudier",
        "windier", "calmer",
        "more_humid", "less_muggy",
    ]
    test_scopes = ["year_round", "winter", "summer"]
    test_intensities = ["noticeably"]

    pinned = ["Tokyo", "Seattle", "Phoenix", "Cairo", "Reykjavik",
              "Singapore", "London", "Sydney"]
    anchor_idxs: list[int] = []
    for n in pinned:
        idx = NAME_TO_IDX.get(n.lower())
        if idx is not None and idx not in anchor_idxs:
            anchor_idxs.append(idx)
    remaining = [i for i in range(n_cities) if i not in set(anchor_idxs)]
    rng.shuffle(remaining)
    anchor_idxs += remaining[: max(0, args.n_anchors - len(anchor_idxs))]
    anchor_idxs = anchor_idxs[: args.n_anchors]

    queries = list(itertools.product(
        anchor_idxs, test_axes, test_scopes, test_intensities
    ))
    print(f"running {len(queries)} synthetic queries "
          f"({len(anchor_idxs)} anchors x {len(test_axes)} axes x "
          f"{len(test_scopes)} scopes)")

    rows = []
    for ai, axis, scope, inten in queries:
        vibe = {"axis": axis, "scope": scope, "intensity": inten}
        out = evaluate(ai, vibe, profiles_scaled, args.top_k)
        rows.append({
            "anchor": CITY_NAMES[ai],
            "axis": axis, "scope": scope, "intensity": inten,
            **out,
            "top_cities": [CITY_NAMES[i] for i in out["top"][:5]],
        })

    cosines = np.array([r["cosine"] for r in rows])
    print()
    print("== overall ==")
    print(f"  mean cosine:        {cosines.mean():+.3f}")
    print(f"  median:             {np.median(cosines):+.3f}")
    print(f"  P25 / P75:          {np.quantile(cosines, 0.25):+.3f} / "
          f"{np.quantile(cosines, 0.75):+.3f}")
    print(f"  fraction >= 0.7:    {(cosines >= 0.7).mean()*100:5.1f}%")
    print(f"  fraction in [0.3,0.7): "
          f"{((cosines >= 0.3) & (cosines < 0.7)).mean()*100:5.1f}%")
    print(f"  fraction <  0.3:    {(cosines <  0.3).mean()*100:5.1f}%")
    print(f"  fraction <  0   :   {(cosines <  0  ).mean()*100:5.1f}%   "
          "(directional failure)")

    print()
    print("== by axis ==")
    print(f"  {'axis':<14} {'mean':>8} {'median':>8} {'<0%':>6} {'n':>5}")
    for axis in test_axes:
        sub = [r["cosine"] for r in rows if r["axis"] == axis]
        if sub:
            arr = np.array(sub)
            print(f"  {axis:<14} {arr.mean():+8.3f} {np.median(arr):+8.3f} "
                  f"{(arr<0).mean()*100:5.1f} {len(sub):>5}")

    print()
    print("== by scope ==")
    print(f"  {'scope':<14} {'mean':>8} {'median':>8} {'<0%':>6} {'n':>5}")
    for scope in test_scopes:
        sub = [r["cosine"] for r in rows if r["scope"] == scope]
        if sub:
            arr = np.array(sub)
            print(f"  {scope:<14} {arr.mean():+8.3f} {np.median(arr):+8.3f} "
                  f"{(arr<0).mean()*100:5.1f} {len(sub):>5}")

    print()
    print(f"== {args.show_examples} best ==")
    for r in sorted(rows, key=lambda r: -r["cosine"])[: args.show_examples]:
        print(f"  cos={r['cosine']:+.2f}  {r['anchor']:<14} + "
              f"{r['axis']}/{r['scope']}: {r['top_cities']}")
    print()
    print(f"== {args.show_examples} worst ==")
    for r in sorted(rows, key=lambda r: r["cosine"])[: args.show_examples]:
        print(f"  cos={r['cosine']:+.2f}  {r['anchor']:<14} + "
              f"{r['axis']}/{r['scope']}: {r['top_cities']}")


if __name__ == "__main__":
    main()
