"""Load the pre-fetched Wikipedia image URLs for every canonical city.

city_images.json is produced by backend/scripts/fetch_city_images.py. It
contains both the original full-resolution URL and a 2560px thumbnail per
city. The tournament tab uses the 2560px thumb — big enough to look sharp
on a split-screen, small enough to not ship 20MB files to the browser.
"""
from __future__ import annotations

import json
from pathlib import Path

_CACHE: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        path = Path(__file__).resolve().parents[1] / "cache" / "city_images.json"
        _CACHE = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _CACHE


def image_for(city_name: str) -> dict:
    """Return {image_url, thumb_url} for a city, or empty dict if unknown."""
    rec = _load().get(city_name) or {}
    if "error" in rec or "thumb_url" not in rec:
        return {"image_url": "", "thumb_url": ""}
    return {
        "image_url": rec.get("original_url", ""),
        "thumb_url": rec.get("thumb_url", ""),
    }
