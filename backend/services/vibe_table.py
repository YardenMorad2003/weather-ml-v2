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
    "warmer":      [(F_TEMP, +1), (F_DEW, +0.5)],
    "colder":      [(F_TEMP, -1), (F_DEW, -0.5)],
    "drier":       [(F_HUM, -1), (F_DEW, -0.5), (F_PRECIP, -0.5)],
    "more_humid":  [(F_HUM, +1), (F_DEW, +0.5)],
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
