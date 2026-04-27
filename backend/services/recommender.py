"""End-to-end recommendation pipeline for text queries.

Reuses the existing repo's profile builder and scaler, but applies
vibe-based deltas in sigma space instead of requiring numeric user prefs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from ..core import weather_ml  # noqa: F401 - sys.path shim
from cities import CITIES  # type: ignore

from .city_resolver import resolve, suggest_close_matches, ResolvedCity
from .nl_parser import parse_query, ParsedQuery
from .vibe_table import apply_vibes
from .reasons import build_reasons, Reason
from .state import get_state

# how much to up-weight the dims the user asked about vs the rest
FOCUS_WEIGHT = 6.0
BASE_WEIGHT = 1.0


def _weighted_distance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Euclidean distance in sigma space.

    Unlike cosine, this measures *how far* the candidate is from the target
    vector — not just angular alignment. An extreme city like Cairo can still
    rank high for "Seattle but sunnier" because the focus weight amplifies
    sunshine-dim alignment, but its distance score reflects the honest
    trade-off: huge gaps on temperature / humidity / precip count against it.
    """
    diff = a - b
    return float(np.sqrt(np.sum(w * diff * diff)))


@dataclass
class CityResult:
    city: str
    country: str
    lat: float
    lon: float
    similarity: float
    reasons: list[Reason]


@dataclass
class AnchorError:
    """Set when the parser extracted an anchor_city the resolver couldn't
    place. The router treats this as a recoverable failure (empty results +
    inline message), not an exception, so the UI can offer suggestions."""
    input: str
    suggestions: list[str]


@dataclass
class RecommendResponse:
    parsed: ParsedQuery
    anchor: Optional[ResolvedCity]
    results: list[CityResult]
    anchor_error: Optional[AnchorError] = None


def recommend_from_text(db: Session, text: str, top_k: int = 10) -> RecommendResponse:
    parsed = parse_query(text)
    st = get_state()
    profiles, scaler, profiles_scaled = st["profiles"], st["scaler"], st["profiles_scaled"]
    feature_std = st["feature_std"]

    anchor: Optional[ResolvedCity] = None
    if parsed.anchor_city:
        anchor = resolve(db, parsed.anchor_city, profiles)
        if anchor is None:
            # The user asked for a specific place we can't locate. Don't fall
            # through to the centroid — that produces honest-looking results
            # ("Hogwarts but sunny" -> Cairo 47%) for a query we should be
            # admitting we couldn't satisfy.
            return RecommendResponse(
                parsed=parsed,
                anchor=None,
                results=[],
                anchor_error=AnchorError(
                    input=parsed.anchor_city,
                    suggestions=suggest_close_matches(db, parsed.anchor_city),
                ),
            )

    if anchor is not None:
        user_raw = anchor.profile.reshape(1, -1)
    else:
        # no anchor in the query at all -> start from dataset centroid
        user_raw = profiles.mean(axis=0, keepdims=True)

    user_scaled = scaler.transform(user_raw)[0]

    vibes = [v.model_dump() for v in parsed.vibes]
    user_scaled, touched = apply_vibes(user_scaled, vibes)

    # weight vector: dims the user modified get FOCUS_WEIGHT, rest get BASE_WEIGHT
    weights = BASE_WEIGHT + (FOCUS_WEIGHT - BASE_WEIGHT) * touched

    dists = np.array([
        _weighted_distance(user_scaled, profiles_scaled[i], weights)
        for i in range(len(profiles_scaled))
    ])

    # Convert distance -> 0-1 similarity for display. Normalize by sqrt of the
    # total weight so the scale is comparable across queries regardless of how
    # many dims are focus-weighted. A random city ends up around exp(-sqrt(2))
    # ~= 0.24; a perfect match -> 1.0; a far-but-ranked-first match (e.g.
    # Cairo for "Seattle but sunnier") honestly shows up at ~0.25-0.35.
    scale = float(np.sqrt(weights.sum()))
    sims = np.exp(-dists / scale)

    # exclude the anchor city from results
    anchor_name = anchor.name.lower() if anchor else None
    anchor_profile = anchor.profile if anchor else None

    order = np.argsort(dists)
    results: list[CityResult] = []
    for i in order:
        if anchor_name and CITIES[i]["name"].lower() == anchor_name:
            continue
        results.append(CityResult(
            city=CITIES[i]["name"],
            country=CITIES[i].get("country", ""),
            lat=CITIES[i]["lat"],
            lon=CITIES[i]["lon"],
            similarity=float(sims[i]),
            reasons=build_reasons(vibes, anchor_profile, profiles[i], feature_std),
        ))
        if len(results) >= top_k:
            break

    return RecommendResponse(parsed=parsed, anchor=anchor, results=results)
