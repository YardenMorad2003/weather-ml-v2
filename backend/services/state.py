"""Shared pipeline state: hemisphere-normalized profiles, scaler, PCA.

Built once per process on first access. Recommender and PCA endpoints
share this so we don't rebuild the 102-city profile matrix twice.
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

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
        pca = PCA(n_components=3).fit(profiles_scaled)
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

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.explained_variance_ratio_ = []
        self.components_ = []

    def fit(self, data):
        if type(data) != np.array:
            data = np.array(data)
        
        # first get covariance matrix
        covariance_matrix = np.cov(data, rowvar=False)

        # get eigenvalues and eigenvectors of the covariance matrix
        eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)

        # pair eigenvalue to eigenvector and then sort by largest eigenvalues
        arr = [(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        arr = sorted(arr, key = lambda x: -1 * x[0])

        # de-pair
        evals_desc = np.array([x[0] for x in arr])
        evecs_desc = np.array([x[1] for x in arr])

        # add the explained variance ratios
        total_variance = np.sum(evals_desc)
        self.explained_variance_ratio_ = [round(var / total_variance, 4) for var in evals_desc]
        
        # store eigenvectors
        self.components_ = evecs_desc[:self.n_components].T
    
    def transform(self, data):
        if len(self.components_) == 0:
            raise RuntimeError("You have not fit a dataset yet.")

        return data @ self.components_