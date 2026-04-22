"""Per-result explainers.

For each parsed vibe, compute a factual diff between the candidate city and
the anchor (or absolute reading if no anchor). Output is purely a function of
the raw profiles — no LLM — so users can audit *why* a city surfaced.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vibe_table import (
    SCOPE_MONTHS,
    F_TEMP, F_HUM, F_PRECIP, F_WIND, F_CLEAR,
)

# axis -> (display name, feature index, unit, scale factor, decimal places)
_AXIS_PRIMARY: dict[str, tuple[str, int, str, float, int]] = {
    "warmer":     ("Temp",       F_TEMP,   "°C",    1.0,   1),
    "colder":     ("Temp",       F_TEMP,   "°C",    1.0,   1),
    "drier":      ("Humidity",   F_HUM,    "%",     1.0,   0),
    "more_humid": ("Humidity",   F_HUM,    "%",     1.0,   0),
    "less_muggy": ("Humidity",   F_HUM,    "%",     1.0,   0),
    "wetter":     ("Precip",     F_PRECIP, " mm/h", 1.0,   2),
    "less_rainy": ("Precip",     F_PRECIP, " mm/h", 1.0,   2),
    "sunnier":    ("Clear sky",  F_CLEAR,  "%",     100.0, 0),
    "cloudier":   ("Clear sky",  F_CLEAR,  "%",     100.0, 0),
    "windier":    ("Wind",       F_WIND,   " km/h", 1.0,   1),
    "calmer":     ("Wind",       F_WIND,   " km/h", 1.0,   1),
}

_AXIS_DIRECTION: dict[str, int] = {
    "warmer": +1, "colder": -1,
    "more_humid": +1, "drier": -1, "less_muggy": -1,
    "wetter": +1, "less_rainy": -1,
    "sunnier": +1, "cloudier": -1,
    "windier": +1, "calmer": -1,
}

_SCOPE_LABEL = {
    "winter": "cold season",
    "summer": "warm season",
    "year_round": "year-round",
}

# the 5 features a user can read without climatology background.
# dewpoint, cloud, and pressure are omitted (redundant with humidity/sunshine,
# or not human-meaningful).
_KEY_FEATURES: list[tuple[str, int, str, float, int]] = [
    ("Temperature",   F_TEMP,   "°C",    1.0,   1),
    ("Humidity",      F_HUM,    "%",     1.0,   0),
    ("Precipitation", F_PRECIP, " mm/h", 1.0,   2),
    ("Wind",          F_WIND,   " km/h", 1.0,   1),
    ("Sunshine",      F_CLEAR,  "%",     100.0, 0),
]

# which feature a vibe axis primarily drives; used to skip that feature in
# the supporting-reasons list so we don't show a primary reason twice.
_AXIS_FEATURE_IDX: dict[str, int] = {
    "warmer": F_TEMP, "colder": F_TEMP,
    "milder": F_TEMP, "more_extreme": F_TEMP,
    "more_seasonal": F_TEMP, "less_seasonal": F_TEMP,
    "drier": F_HUM, "more_humid": F_HUM, "less_muggy": F_HUM,
    "wetter": F_PRECIP, "less_rainy": F_PRECIP,
    "sunnier": F_CLEAR, "cloudier": F_CLEAR,
    "windier": F_WIND, "calmer": F_WIND,
}


def _annual_mean(profile: np.ndarray, feat_idx: int) -> float:
    m = profile.reshape(12, 8)
    return float(m[:, feat_idx].mean())


def _supporting_reasons(
    vibes: list[dict],
    anchor_profile: np.ndarray,
    candidate_profile: np.ndarray,
    feature_std: np.ndarray,
) -> list[Reason]:
    """Annual-mean diffs on non-vibed key features, sorted by closeness."""
    vibed_feats = {
        _AXIS_FEATURE_IDX[v["axis"]]
        for v in vibes
        if v["axis"] in _AXIS_FEATURE_IDX
    }

    rows: list[tuple[float, str, float, float, str, int]] = []
    for name, feat_idx, unit, scale, prec in _KEY_FEATURES:
        if feat_idx in vibed_feats:
            continue
        anc = _annual_mean(anchor_profile, feat_idx) * scale
        cand = _annual_mean(candidate_profile, feat_idx) * scale
        std = float(feature_std[feat_idx]) * scale
        norm_dist = abs(cand - anc) / (std + 1e-6)
        rows.append((norm_dist, name, cand, anc, unit, prec))

    rows.sort(key=lambda r: r[0])

    reasons: list[Reason] = []
    for norm_dist, name, cand, anc, unit, prec in rows:
        diff = cand - anc
        detail = (
            f"{cand:.{prec}f}{unit} vs {anc:.{prec}f}{unit} "
            f"({diff:+.{prec}f}{unit})"
        )
        reasons.append(Reason(
            label=f"{name} (year-round)",
            detail=detail,
            matched=norm_dist < 0.5,
        ))
    return reasons


@dataclass
class Reason:
    label: str
    detail: str
    matched: bool


def _scope_mean(profile: np.ndarray, feat_idx: int, scope: str) -> float:
    months = SCOPE_MONTHS[scope]
    m = profile.reshape(12, 8)
    return float(m[[mo - 1 for mo in months], feat_idx].mean())


def _temp_range(profile: np.ndarray) -> float:
    m = profile.reshape(12, 8)
    return float(m[:, F_TEMP].max() - m[:, F_TEMP].min())


def build_reasons(
    vibes: list[dict],
    anchor_profile: Optional[np.ndarray],
    candidate_profile: np.ndarray,
    feature_std: Optional[np.ndarray] = None,
) -> list[Reason]:
    reasons: list[Reason] = []

    for v in vibes:
        axis = v["axis"]
        scope = v.get("scope", "year_round")

        if axis in ("more_seasonal", "less_seasonal", "milder", "more_extreme"):
            cand = _temp_range(candidate_profile)
            label = {
                "more_seasonal": "Seasonal swing",
                "less_seasonal": "Seasonal swing",
                "milder": "Temperature swing",
                "more_extreme": "Temperature swing",
            }[axis]
            want = +1 if axis in ("more_seasonal", "more_extreme") else -1
            if anchor_profile is not None:
                anc = _temp_range(anchor_profile)
                diff = cand - anc
                detail = f"{cand:.1f}°C range vs {anc:.1f}°C ({diff:+.1f}°C)"
                matched = (diff * want) > 0
            else:
                detail = f"{cand:.1f}°C range"
                matched = True
            reasons.append(Reason(label=label, detail=detail, matched=matched))
            continue

        spec = _AXIS_PRIMARY.get(axis)
        if spec is None:
            continue
        name, feat_idx, unit, scale, prec = spec

        cand = _scope_mean(candidate_profile, feat_idx, scope) * scale
        label = f"{name} ({_SCOPE_LABEL[scope]})"

        if anchor_profile is not None:
            anc = _scope_mean(anchor_profile, feat_idx, scope) * scale
            diff = cand - anc
            detail = (
                f"{cand:.{prec}f}{unit} vs {anc:.{prec}f}{unit} "
                f"({diff:+.{prec}f}{unit})"
            )
            want = _AXIS_DIRECTION.get(axis, 0)
            matched = (diff * want) > 0 if want != 0 else True
        else:
            detail = f"{cand:.{prec}f}{unit}"
            matched = True

        reasons.append(Reason(label=label, detail=detail, matched=matched))

    if anchor_profile is not None and feature_std is not None:
        reasons.extend(_supporting_reasons(
            vibes, anchor_profile, candidate_profile, feature_std,
        ))

    return reasons
