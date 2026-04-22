"""Patch city_images.json for cities where the default Wikipedia lead image
was wrong (country flag, disambiguation, or too small).

For each override we use a different Wikipedia title that points at a
skyline/landmark article instead of the country/generic page.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.fetch_city_images import fetch, OUT_JSON, REQUEST_DELAY_S  # noqa: E402

OVERRIDES: dict[str, list[str]] = {
    "Portland":       ["Portland, Oregon"],
    "Rochester":      ["Rochester, New York"],
    "San Jose":       ["Downtown San José, Costa Rica", "Paseo Colón", "Teatro Nacional de Costa Rica"],
    "Kingston":       ["New Kingston", "Downtown Kingston", "Emancipation Park, Jamaica"],
    "Singapore":      ["Marina Bay, Singapore", "Downtown Core", "Central Area, Singapore"],
    "Hong Kong":      ["Victoria Harbour", "Central, Hong Kong", "Hong Kong Island"],
    "Port-au-Prince": ["Downtown Port-au-Prince", "Champ de Mars, Port-au-Prince"],
    "Murmansk":       ["Murmansk"],
    "Yellowknife":    ["Yellowknife"],
    "Palermo":        ["Palermo", "Historic Centre of Palermo"],
    "Antananarivo":   ["Antananarivo"],
    "Port Louis":     ["Port Louis"],
}


def main() -> None:
    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    for city, titles in OVERRIDES.items():
        print(f"[{city}] trying overrides: {titles}")
        best = None
        best_title = None
        for t in titles:
            try:
                rec = fetch(t)
            except Exception as exc:  # noqa: BLE001
                print(f"  error on '{t}': {exc}")
                continue
            time.sleep(REQUEST_DELAY_S)
            if rec is None or rec.get("disambiguation"):
                print(f"  '{t}' -> miss/disambig")
                continue
            w = rec.get("original_width", 0)
            print(f"  '{t}' -> {rec.get('file_name')} ({w}px)")
            if best is None or w > best.get("original_width", 0):
                best = rec
                best_title = t
        if best is None:
            print(f"  no override worked for {city}, leaving prior entry")
            continue
        best["override_title"] = best_title
        best["issues"] = []  # re-evaluate by filename? skip, manual review
        data[city] = best

    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPatched {OUT_JSON}")


if __name__ == "__main__":
    main()
