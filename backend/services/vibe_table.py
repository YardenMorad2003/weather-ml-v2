"""Vibe vocabulary and vibe -> feature-space delta lookup.

The city embedding is 96-d: 12 months x 8 features
  [temp, humidity, dewpoint, precip, cloud, pressure, wind, clear_sky]
Deltas are applied in StandardScaler space (sigma units), so "noticeably
warmer" = +1.0 on every temp dimension within the scoped months.

This table is the hand-tuned knob. Tune after looking at real results.
"""
from __future__ import annotations
from typing import Literal
import numpy as np

# --- vocabulary --------------------------------------------------------------

VIBE_AXES = [
    "warmer", "colder", "milder", "more_extreme",
    "more_seasonal", "less_seasonal",
    "drier", "more_humid", "less_muggy", "wetter", "less_rainy",
    "sunnier", "cloudier",
    "windier", "calmer",
]

SCOPES = ["winter", "summer", "year_round"]
INTENSITIES = ["slightly", "noticeably", "much"]

INTENSITY_SIGMA = {"slightly": 1.2, "noticeably": 2.5, "much": 4.0}

# feature index within the 8-tuple per month
F_TEMP, F_HUM, F_DEW, F_PRECIP, F_CLOUD, F_PRESS, F_WIND, F_CLEAR = range(8)

# months (1-12) per scope. Northern-hemisphere-centric; fine for v1.
SCOPE_MONTHS = {
    "winter": [12, 1, 2],
    "summer": [6, 7, 8],
    "year_round": list(range(1, 13)),
}

# axis -> list of (feature_idx, sign). Applied to every scoped month.
AXIS_FEATURES: dict[str, list[tuple[int, float]]] = {
    "warmer":      [(F_TEMP, +1), (F_DEW, +0.8)],
    "colder":      [(F_TEMP, -1), (F_DEW, -0.8)],
    "drier":       [(F_HUM, -1), (F_PRECIP, -0.5)],
    "more_humid":  [(F_HUM, +1)],
    "less_muggy":  [(F_HUM, -1), (F_DEW, -1)],
    "wetter":      [(F_PRECIP, +1), (F_CLOUD, +0.5), (F_CLEAR, -0.5)],
    "less_rainy":  [(F_PRECIP, -1), (F_CLOUD, -0.3)],
    "sunnier":     [(F_CLEAR, +1), (F_CLOUD, -1)],
    "cloudier":    [(F_CLEAR, -1), (F_CLOUD, +1)],
    "windier":     [(F_WIND, +1)],
    "calmer":      [(F_WIND, -1)],
    # seasonality & mildness handled specially below
    "more_seasonal":  [],
    "less_seasonal":  [],
    "milder":         [],
    "more_extreme":   [],
}


# Pairs handled by special-branch logic in apply_vibes (not via AXIS_FEATURES).
# detect_vibe_conflicts() can't see these via the feature-overlap check, so
# list them explicitly.
_SEMANTIC_OPPOSITES: list[frozenset[str]] = [
    frozenset({"milder", "more_extreme"}),
    frozenset({"more_seasonal", "less_seasonal"}),
]


def detect_vibe_conflicts(vibes: list[dict]) -> list[dict]:
    """Find pairs of vibes that genuinely contradict each other. Two flavors:

      1. Semantic opposites (milder/more_extreme, more_seasonal/less_seasonal)
         — the special-branch axes that don't expose couplings via AXIS_FEATURES.
      2. Feature-level opposition where EITHER both vibes touch the same
         feature as their *primary* coupling (the leading entry, max
         magnitude) with opposite signs, OR ≥2 features oppose at once.

    Single-secondary-feature opposition (e.g. less_muggy's dewpoint at -1 vs
    warmer's dewpoint at +0.8) is *not* flagged — the eval shows those compound
    queries score near +0.80, well within the working range. This rule fires
    only on pairs where the σ-eval shows compound cosines drop below ~+0.6.
    """
    conflicts: list[dict] = []
    for i in range(len(vibes)):
        for j in range(i + 1, len(vibes)):
            v1, v2 = vibes[i], vibes[j]
            ax1, ax2 = v1["axis"], v2["axis"]

            if frozenset({ax1, ax2}) in _SEMANTIC_OPPOSITES:
                conflicts.append({
                    "axis_a": ax1, "axis_b": ax2, "reason": "semantic_opposite",
                })
                continue

            f1 = AXIS_FEATURES.get(ax1, [])
            f2 = AXIS_FEATURES.get(ax2, [])
            if not f1 or not f2:
                continue

            months1 = set(SCOPE_MONTHS[v1.get("scope", "year_round")])
            months2 = set(SCOPE_MONTHS[v2.get("scope", "year_round")])
            if not (months1 & months2):
                continue

            # Find all opposed features and tag whether each is primary
            # (leading entry — the axis's main feature) for either axis.
            primary1, primary2 = f1[0][0], f2[0][0]
            opposed_features: list[tuple[int, bool]] = []
            for feat1, sign1 in f1:
                for feat2, sign2 in f2:
                    if feat1 == feat2 and (sign1 * sign2) < 0:
                        is_primary = (feat1 == primary1) or (feat1 == primary2)
                        opposed_features.append((feat1, is_primary))

            if not opposed_features:
                continue

            primary_conflict = any(is_p for _, is_p in opposed_features)
            multi_conflict = len(set(f for f, _ in opposed_features)) >= 2

            if primary_conflict or multi_conflict:
                conflicts.append({
                    "axis_a": ax1, "axis_b": ax2,
                    "feature_indices": sorted(set(f for f, _ in opposed_features)),
                    "reason": "primary_opposed" if primary_conflict else "multi_opposed",
                })
    return conflicts


def apply_vibes(scaled_vec: np.ndarray, vibes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Apply vibes to a StandardScaler'd 96-d vector.

    Returns (modified_vec, touched_mask) where touched_mask is a (96,) float
    array marking which (month, feature) dims a vibe acted on. The caller
    uses this mask to weight the ranking so the user's intent dominates.
    """
    out = scaled_vec.copy().reshape(12, 8)
    touched = np.zeros((12, 8), dtype=np.float32)

    def _touch(month_idx: int, feat_idx: int):
        touched[month_idx, feat_idx] = 1.0

    for v in vibes:
        axis = v["axis"]
        scope = v.get("scope", "year_round")
        intensity = v.get("intensity", "noticeably")
        mag = INTENSITY_SIGMA[intensity]
        months = SCOPE_MONTHS[scope]

        if axis in ("more_seasonal", "less_seasonal"):
            temp_col = out[:, F_TEMP]
            mean = temp_col.mean()
            factor = (1 + mag) if axis == "more_seasonal" else max(0.0, 1 - mag * 0.5)
            out[:, F_TEMP] = mean + (temp_col - mean) * factor
            for m in range(12):
                _touch(m, F_TEMP)
            continue

        if axis in ("milder", "more_extreme"):
            if axis == "milder":
                mean = out[:, F_TEMP].mean()
                out[:, F_TEMP] = mean + (out[:, F_TEMP] - mean) * max(0.0, 1 - mag * 0.5)
                for m in range(12):
                    _touch(m, F_TEMP)
            else:
                for m in SCOPE_MONTHS["winter"]:
                    out[m - 1, F_TEMP] -= mag
                    _touch(m - 1, F_TEMP)
                for m in SCOPE_MONTHS["summer"]:
                    out[m - 1, F_TEMP] += mag
                    _touch(m - 1, F_TEMP)
            continue

        for feat_idx, sign in AXIS_FEATURES[axis]:
            for m in months:
                out[m - 1, feat_idx] += sign * mag
                _touch(m - 1, feat_idx)

    return out.reshape(96), touched.reshape(96)
