"""Resolve a (possibly misspelled) city name to a 96-d weather profile.

Resolution order:
  1. Exact (normalized) match against the canonical 102 cities
  2. Fuzzy match (difflib) against the canonical 102 cities
  3. Fuzzy match against fetched_cities cache in Postgres
  4. Open-Meteo geocoding -> fetch hourly history -> make_city_profile -> cache
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from sqlalchemy.orm import Session

from ..core import weather_ml  # noqa: F401 - sets up sys.path
from cities import CITIES  # type: ignore
from data import fetch_history_chunked  # type: ignore
from features import make_city_profile  # type: ignore

from ..db.models import FetchedCity
from .profile import phase_align

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FUZZY_CUTOFF = 0.75
# Kept in sync with climatology.py defaults — any change here should match
# `build_all_profiles_multi_year` so canonical and on-demand cities draw from
# the same climatology window.
HISTORY_START = "2023-01-01"
HISTORY_END = "2024-12-31"


def _norm(s: str) -> str:
    return s.strip().lower()


@dataclass
class ResolvedCity:
    name: str
    country: str
    lat: float
    lon: float
    profile: np.ndarray  # 96-d float32, in the same units as canonical profiles
    source: str  # "canonical" | "cache" | "open-meteo"


def _canonical_index() -> dict[str, int]:
    return {_norm(c["name"]): i for i, c in enumerate(CITIES)}


def _fuzzy_pick(query: str, choices: list[str]) -> Optional[str]:
    matches = difflib.get_close_matches(query, choices, n=1, cutoff=FUZZY_CUTOFF)
    return matches[0] if matches else None


def _profile_for_canonical(idx: int, all_profiles: np.ndarray) -> ResolvedCity:
    c = CITIES[idx]
    return ResolvedCity(
        name=c["name"],
        country=c.get("country", ""),
        lat=c["lat"],
        lon=c["lon"],
        profile=all_profiles[idx].astype(np.float32),
        source="canonical",
    )


def _load_cached(db: Session, key: str) -> Optional[ResolvedCity]:
    row = db.query(FetchedCity).filter(FetchedCity.key == key).one_or_none()
    if row is None:
        return None
    prof = np.frombuffer(row.profile, dtype=np.float32).copy()
    # phase_align is idempotent on already-aligned rows, so this also repairs
    # legacy blobs written under the old hemisphere_normalize.
    prof = phase_align(prof).astype(np.float32)
    return ResolvedCity(
        name=row.name, country=row.country, lat=row.lat, lon=row.lon,
        profile=prof, source="cache",
    )


def _fuzzy_cached(db: Session, query_norm: str) -> Optional[ResolvedCity]:
    rows = db.query(FetchedCity).all()
    if not rows:
        return None
    keys = [r.key for r in rows]
    hit = _fuzzy_pick(query_norm, keys)
    if hit is None:
        return None
    return _load_cached(db, hit)


def _geocode(name: str) -> Optional[dict]:
    try:
        r = requests.get(GEOCODE_URL, params={"name": name, "count": 1}, timeout=10)
        r.raise_for_status()
        results = r.json().get("results") or []
        return results[0] if results else None
    except requests.RequestException:
        return None


def _fetch_and_cache(db: Session, key: str, name: str) -> Optional[ResolvedCity]:
    geo = _geocode(name)
    if geo is None:
        return None

    lat = float(geo["latitude"])
    lon = float(geo["longitude"])
    canonical_name = geo.get("name", name)
    country = geo.get("country", "")

    try:
        df = fetch_history_chunked(lat, lon, HISTORY_START, HISTORY_END, freq="hourly")
        profile = make_city_profile(df).astype(np.float32)
        profile = phase_align(profile).astype(np.float32)
    except Exception:
        return None

    row = FetchedCity(
        key=key, name=canonical_name, country=country,
        lat=lat, lon=lon, profile=profile.tobytes(),
    )
    db.add(row)
    db.commit()

    return ResolvedCity(
        name=canonical_name, country=country, lat=lat, lon=lon,
        profile=profile, source="open-meteo",
    )


def resolve(db: Session, user_input: str, all_profiles: np.ndarray) -> Optional[ResolvedCity]:
    """Resolve a user-provided city name to a weather profile.

    `all_profiles` is the canonical 102-city array returned by
    recommend.build_all_profiles(), passed in so we don't rebuild it here.
    """
    if not user_input:
        return None

    key = _norm(user_input)
    canon = _canonical_index()

    # 1. exact canonical
    if key in canon:
        return _profile_for_canonical(canon[key], all_profiles)

    # 2. fuzzy canonical
    hit = _fuzzy_pick(key, list(canon.keys()))
    if hit is not None:
        return _profile_for_canonical(canon[hit], all_profiles)

    # 3. exact cache
    cached = _load_cached(db, key)
    if cached is not None:
        return cached

    # 4. fuzzy cache
    cached = _fuzzy_cached(db, key)
    if cached is not None:
        return cached

    # 5. geocode + fetch + cache
    return _fetch_and_cache(db, key, user_input)
