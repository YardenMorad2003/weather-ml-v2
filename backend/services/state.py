"""Shared pipeline state: hemisphere-normalized profiles, scaler, PCA.

Built once per process on first access. Recommender and PCA endpoints
share this so we don't rebuild the 102-city profile matrix twice.
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..core import weather_ml  # noqa: F401
from cities import CITIES  # type: ignore

from .climatology import build_all_profiles_multi_year
from .profile import phase_align

_cache: dict = {}


def get_state():
    if "profiles" not in _cache:
        raw = build_all_profiles_multi_year()
        profiles = np.stack([phase_align(raw[i]) for i in range(len(CITIES))])
        scaler = StandardScaler().fit(profiles)
        profiles_scaled = scaler.transform(profiles)
        pca = PCA(n_components=3, random_state=42).fit(profiles_scaled)
        # per-feature std of annual means across all cities, for normalizing
        # "is this feature close to the anchor?" comparisons in reasons.py
        annual_per_feat = profiles.reshape(len(CITIES), 12, 8).mean(axis=1)
        feature_std = annual_per_feat.std(axis=0)
        _cache.update({
            "profiles": profiles,
            "scaler": scaler,
            "profiles_scaled": profiles_scaled,
            "pca": pca,
            "profiles_proj": pca.transform(profiles_scaled),
            "feature_std": feature_std,
        })
    return _cache
