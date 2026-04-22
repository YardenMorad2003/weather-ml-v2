"""Shim that exposes the vendored weather-ml modules as flat imports.

The four files we use (cities, data, features) live under
`backend/vendor/weather_ml/`. We prepend that directory to sys.path on first
import so callers can keep writing `from cities import CITIES` etc. without
threading a package path through every import.
"""
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "weather_ml"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import cities  # noqa: E402
import data  # noqa: E402
import features  # noqa: E402

__all__ = ["cities", "data", "features"]
