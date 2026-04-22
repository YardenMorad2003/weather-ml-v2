"""Split-screen tournament: user picks between two cities for N rounds,
and each pick nudges a running anchor vector in sigma space. After the
last round we rank all 230 cities against the learned anchor using the
same weighted-Euclidean recommender the text query uses.

Design choices:
- Stateless: the client re-sends the full history each request. No session
  table, no restart concerns, trivially horizontally scalable.
- Anchor update is an EMA toward the chosen city's sigma-space profile
  with alpha = 0.35. Picks matter more early (when the anchor is the
  centroid) than late (when it's already pulled toward the user's taste).
- Pair selection narrows over rounds: early rounds show two candidates
  with a wider rank gap (exploration), late rounds show two neighbors
  (refinement). This converges on the user's preferred region faster
  than random pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import weather_ml  # noqa: F401
from cities import CITIES  # type: ignore

from .images import image_for
from .recommender import _weighted_distance, FOCUS_WEIGHT, BASE_WEIGHT  # noqa: F401
from .state import get_state

TOTAL_ROUNDS = 10
EMA_ALPHA = 0.35


@dataclass
class CityStats:
    temp_c: float
    humidity_pct: float
    precip_mm: float
    sun_pct: float


@dataclass
class PairCity:
    name: str
    country: str
    lat: float
    lon: float
    image_url: str
    thumb_url: str
    stats: CityStats


@dataclass
class PairOut:
    round: int
    total_rounds: int
    pair: list[PairCity]


@dataclass
class FinalOut:
    rounds_completed: int
    picked: list[str]
    results: list  # list[CityResult] from recommender


HistoryItem = dict  # {"shown": [str, str], "picked": str}


def _city_index(name: str) -> Optional[int]:
    key = name.strip().lower()
    for i, c in enumerate(CITIES):
        if c["name"].lower() == key:
            return i
    return None


F_TEMP, F_HUM, _, F_PRECIP, _, _, _, F_CLEAR = range(8)


def _annual_stats(profile: np.ndarray) -> CityStats:
    m = profile.reshape(12, 8).mean(axis=0)
    return CityStats(
        temp_c=float(m[F_TEMP]),
        humidity_pct=float(m[F_HUM]),
        precip_mm=float(m[F_PRECIP]),
        sun_pct=float(m[F_CLEAR]) * 100.0,
    )


def _make_pair_city(idx: int) -> PairCity:
    c = CITIES[idx]
    img = image_for(c["name"])
    profiles = get_state()["profiles"]
    return PairCity(
        name=c["name"],
        country=c.get("country", ""),
        lat=c["lat"],
        lon=c["lon"],
        image_url=img["image_url"],
        thumb_url=img["thumb_url"],
        stats=_annual_stats(profiles[idx]),
    )


def _compute_anchor(history: list[HistoryItem], profiles_scaled: np.ndarray) -> np.ndarray:
    """Replay picks from the centroid forward with an EMA update."""
    anchor = profiles_scaled.mean(axis=0).copy()
    for item in history:
        idx = _city_index(item.get("picked", ""))
        if idx is None:
            continue
        anchor = (1.0 - EMA_ALPHA) * anchor + EMA_ALPHA * profiles_scaled[idx]
    return anchor


def _used_indices(history: list[HistoryItem]) -> set[int]:
    """Every city that has been shown, whether picked or not — don't repeat."""
    used: set[int] = set()
    for item in history:
        for name in item.get("shown", []):
            idx = _city_index(name)
            if idx is not None:
                used.add(idx)
    return used


def _pick_pair(
    anchor: np.ndarray,
    profiles_scaled: np.ndarray,
    used: set[int],
    round_num: int,
    seed: Optional[int] = None,
) -> tuple[int, int]:
    """Return two city indices to show this round.

    Rank unused cities by distance to the current anchor. Pick a low-rank
    city (close-to-anchor) and a higher-rank city (further). The gap
    narrows over rounds. When `seed` is given, jitter the choice within
    small windows so each game shows different openings — same seed +
    same history is still deterministic (for re-renders).
    """
    n = len(profiles_scaled)
    weights = np.ones_like(anchor)
    dists = np.array([
        _weighted_distance(anchor, profiles_scaled[i], weights) if i not in used else np.inf
        for i in range(n)
    ])
    order = np.argsort(dists)
    order = [int(i) for i in order if not np.isinf(dists[i])]
    if len(order) < 2:
        raise ValueError("not enough unused cities for a pair")

    spread = max(2, 22 - round_num * 2)  # r1=20, r5=12, r10=2

    if seed is not None:
        rng = np.random.RandomState((seed + round_num * 9973) % (2**31))
        low_pool = min(8, len(order))  # random low from top 8 nearest
        low_idx = int(rng.randint(0, low_pool))
        # pick high from a window around low_idx + spread, still within bounds
        hi_lo = min(low_idx + max(2, spread - 3), len(order) - 1)
        hi_hi = min(low_idx + spread + 3, len(order))
        if hi_hi <= hi_lo:
            hi_hi = hi_lo + 1
        high_idx = int(rng.randint(hi_lo, hi_hi))
    else:
        low_idx = max(0, min(round_num - 1, len(order) - 2))
        high_idx = min(low_idx + spread, len(order) - 1)

    return order[low_idx], order[high_idx]


def next_pair(
    history: list[HistoryItem],
    seed: Optional[int] = None,
) -> PairOut:
    st = get_state()
    profiles_scaled = st["profiles_scaled"]

    round_num = len(history) + 1
    if round_num > TOTAL_ROUNDS:
        raise ValueError(f"tournament already complete ({TOTAL_ROUNDS} rounds)")

    anchor = _compute_anchor(history, profiles_scaled)
    used = _used_indices(history)
    i, j = _pick_pair(anchor, profiles_scaled, used, round_num, seed=seed)

    return PairOut(
        round=round_num,
        total_rounds=TOTAL_ROUNDS,
        pair=[_make_pair_city(i), _make_pair_city(j)],
    )


def finalize(history: list[HistoryItem], top_k: int = 10):
    """After the last pick, rank all 230 cities vs the learned anchor."""
    from .recommender import CityResult
    from .reasons import build_reasons

    st = get_state()
    profiles = st["profiles"]
    profiles_scaled = st["profiles_scaled"]
    scaler = st["scaler"]
    feature_std = st["feature_std"]

    anchor_scaled = _compute_anchor(history, profiles_scaled)
    # unweighted Euclidean in sigma space for the final rank — no vibes,
    # user expressed preference via picks
    weights = np.ones_like(anchor_scaled)
    dists = np.array([
        _weighted_distance(anchor_scaled, profiles_scaled[i], weights)
        for i in range(len(profiles_scaled))
    ])
    scale = float(np.sqrt(weights.sum()))
    sims = np.exp(-dists / scale)

    picked_names = [item.get("picked", "") for item in history]
    picked_set = {n.lower() for n in picked_names if n}
    # exclude cities the user already saw? no — user wants the verdict,
    # including how close their picks themselves rank

    order = np.argsort(dists)
    results: list[CityResult] = []
    # inverse-transform the anchor back to raw space so reasons.py can diff
    # against it on a per-feature basis
    anchor_raw = scaler.inverse_transform(anchor_scaled.reshape(1, -1))[0]

    for i in order:
        results.append(CityResult(
            city=CITIES[i]["name"],
            country=CITIES[i].get("country", ""),
            lat=CITIES[i]["lat"],
            lon=CITIES[i]["lon"],
            similarity=float(sims[i]),
            reasons=build_reasons([], anchor_raw, profiles[i], feature_std),
        ))
        if len(results) >= top_k:
            break

    return FinalOut(
        rounds_completed=len(history),
        picked=picked_names,
        results=results,
    )
