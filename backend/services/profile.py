"""Profile transformations shared between recommender and resolver."""
from __future__ import annotations
import numpy as np


def phase_align(profile: np.ndarray) -> np.ndarray:
    """Roll the 12-month profile so each city's coldest temp month sits at
    index 0. Replaces the old hemisphere_normalize (which assumed NH's
    coldest was always Jan and blindly rolled SH by 6).

    Works for every latitude:
      - NH city with Jan coldest  -> no change (already at 0)
      - NH city with Feb coldest  -> rolled by -1 so Feb -> index 0
      - SH city with Jul coldest  -> rolled by -6 so Jul -> index 0
      - tropical city with weak seasonality -> argmin is noisy but values
        are near-constant, so the roll is near-cosmetic

    Idempotent: applying it to an already-aligned profile is a no-op, so
    it's safe to run on legacy hemisphere-normalized cache rows.

    profile: (96,) float array = 12 months x 8 features, flattened.
    """
    m = profile.reshape(12, 8)
    cold_idx = int(np.argmin(m[:, 0]))  # feature 0 = temperature
    return np.roll(m, shift=-cold_idx, axis=0).reshape(96)
