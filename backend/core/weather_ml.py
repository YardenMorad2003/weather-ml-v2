"""Shim that exposes the existing weather-ml repo as importable modules.

Adds WEATHER_ML_PATH to sys.path on first import so we can reuse
features.py / recommend.py / data.py / cities.py without forking.
"""
import sys
from pathlib import Path

from .config import settings

_path = Path(settings.weather_ml_path).resolve()
if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

import cities  # noqa: E402
import data  # noqa: E402
import features  # noqa: E402
import recommend  # noqa: E402

__all__ = ["cities", "data", "features", "recommend"]
