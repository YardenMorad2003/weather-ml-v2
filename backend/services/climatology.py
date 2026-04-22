"""Multi-year climatology builder for the canonical 102 cities.

Replaces weather-ml's `build_all_profiles()`, which is hardcoded to a single
year (2024) and has a cache key that ignores the year range — so asking it
for 5 years of data silently returns the stale single-year cache.

We call `fetch_history_chunked` + `make_city_profile` ourselves. Because
`make_city_profile` averages across every month in the input DataFrame, a
multi-year hourly range produces a proper climatology (monthly means across
all years) in one call.

The resulting (n_cities, 96) matrix is cached to `backend/cache/*.npz` keyed
by the exact year range. First build is slow (network + sibling repo's chunk
cache); subsequent startups load from npz instantly.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

from ..core import weather_ml  # noqa: F401 - sys.path shim
from cities import CITIES  # type: ignore
from data import fetch_history_chunked  # type: ignore
from features import make_city_profile  # type: ignore

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _cache_path(year_start: str, year_end: str) -> Path:
    return _CACHE_DIR / f"profiles_{year_start}_{year_end}.npz"


def build_all_profiles_multi_year(
    year_start: str = "2023-01-01",
    year_end: str = "2024-12-31",
) -> np.ndarray:
    """Return (n_cities, 96) float32 of raw profiles averaged over the range.

    Raw = not hemisphere- or phase-aligned. Callers are expected to apply
    `phase_align` before using for similarity / PCA.
    """
    cache = _cache_path(year_start, year_end)
    if cache.exists():
        data = np.load(cache)
        arr = data["profiles"]
        if arr.shape[0] == len(CITIES):
            return arr

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = np.zeros((len(CITIES), 96), dtype=np.float32)
    total = len(CITIES)
    print(
        f"[climatology] building {total}-city profiles for {year_start}..{year_end}. "
        f"First run hits Open-Meteo; subsequent runs load from cache.",
        file=sys.stderr,
        flush=True,
    )
    for i, c in enumerate(CITIES):
        df = fetch_history_chunked(
            c["lat"], c["lon"], year_start, year_end, freq="hourly"
        )
        profiles[i] = make_city_profile(df).astype(np.float32)
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(
                f"[climatology] {i + 1}/{total} cities built",
                file=sys.stderr,
                flush=True,
            )

    np.savez(cache, profiles=profiles)
    return profiles
