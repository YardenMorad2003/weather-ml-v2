"""Detailed inspection mode for the sigma-space eval.

Picks a handful of illustrative (anchor, vibe) queries and shows:
  - top-10 cities returned by the classical ranker
  - expected delta summary (per touched feature, averaged across touched months)
  - observed delta summary (top-K centroid - anchor on the same feature/months)
  - cosine score in the touched-dim subspace

Use this to sanity-check what the headline cosine number actually means.
"""
from __future__ import annotations

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
FEATURE_NAMES = [
    "temp", "humidity", "dewpoint", "precip",
    "cloud", "pressure", "wind", "clear_sky",
]

CITY_NAMES = [c["name"] for c in CITIES]
NAME_TO_IDX = {n.lower(): i for i, n in enumerate(CITY_NAMES)}


def summarize_delta(delta_96: np.ndarray, touched_96: np.ndarray) -> str:
    """Return a string like 'temp +2.50sigma (12 mo), dewpoint +1.25sigma (12 mo)'."""
    delta = delta_96.reshape(12, 8)
    touched = touched_96.reshape(12, 8)
    parts = []
    for f in range(8):
        mask = touched[:, f] > 0
        if not mask.any():
            continue
        avg = float(delta[mask, f].mean())
        n_months = int(mask.sum())
        sign = "+" if avg >= 0 else "-"
        parts.append(f"{FEATURE_NAMES[f]} {sign}{abs(avg):.2f}sigma ({n_months} mo)")
    return ", ".join(parts) if parts else "(no touched dims)"


def evaluate_verbose(anchor: str, vibe: dict, profiles_scaled, top_k=10):
    ai = NAME_TO_IDX[anchor.lower()]
    user_scaled = profiles_scaled[ai].copy()
    user_modified, touched = apply_vibes(user_scaled, [vibe])
    expected_delta = (user_modified - user_scaled) * touched

    weights = BASE_WEIGHT + (FOCUS_WEIGHT - BASE_WEIGHT) * touched
    diffs = profiles_scaled - user_modified[None, :]
    dists = np.sqrt((weights[None, :] * diffs * diffs).sum(axis=1))

    order = np.argsort(dists).tolist()
    top = [i for i in order if i != ai][:top_k]

    observed_centroid = profiles_scaled[top].mean(axis=0)
    observed_delta = (observed_centroid - user_scaled) * touched

    exp_norm = float(np.linalg.norm(expected_delta))
    obs_norm = float(np.linalg.norm(observed_delta))
    cos = float(np.dot(expected_delta, observed_delta)
                / (exp_norm * obs_norm + EPS))

    label = f"\"{anchor} + {vibe['axis']}/{vibe['scope']}/{vibe['intensity']}\""
    print("=" * 72)
    print(label)
    print(f"  cosine: {cos:+.3f}   |expected|={exp_norm:.2f}sigma   "
          f"|observed|={obs_norm:.2f}sigma")
    print(f"  expected: {summarize_delta(expected_delta, touched)}")
    print(f"  observed: {summarize_delta(observed_delta, touched)}")
    print(f"  top-{top_k}: " + ", ".join(CITY_NAMES[i] for i in top))


def main():
    print("loading state...", flush=True)
    st = get_state()
    profiles_scaled = st["profiles_scaled"]
    print(f"  {len(profiles_scaled)} cities, dim {profiles_scaled.shape[1]}\n")

    cases = [
        ("Reykjavik", "warmer",     "year_round", "noticeably"),
        ("Phoenix",   "colder",     "summer",     "noticeably"),
        ("Tokyo",     "drier",      "summer",     "noticeably"),
        ("London",    "sunnier",    "winter",     "noticeably"),
        ("Seattle",   "drier",      "year_round", "noticeably"),
        ("Singapore", "more_humid", "year_round", "noticeably"),
        ("Muscat",    "warmer",     "summer",     "noticeably"),
        ("Lusaka",    "sunnier",    "winter",     "noticeably"),
        ("Tokyo",     "warmer",     "winter",     "slightly"),
        ("Tokyo",     "warmer",     "winter",     "much"),
    ]
    for anchor, axis, scope, inten in cases:
        evaluate_verbose(
            anchor,
            {"axis": axis, "scope": scope, "intensity": inten},
            profiles_scaled,
            top_k=10,
        )
        print()


if __name__ == "__main__":
    main()
