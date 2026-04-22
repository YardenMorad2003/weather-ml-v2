"""Fetch high-resolution Wikipedia lead images for the canonical city list.

Writes:
  backend/cache/city_images.json        — per-city URLs + metadata
  backend/cache/city_images_preview.html — visual review grid, bad candidates flagged

Run from repo root:
  python backend/scripts/fetch_city_images.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.vendor.weather_ml.cities import CITIES  # noqa: E402

CACHE_DIR = REPO_ROOT / "backend" / "cache"
OUT_JSON = CACHE_DIR / "city_images.json"
OUT_HTML = CACHE_DIR / "city_images_preview.html"

API = "https://en.wikipedia.org/w/api.php"
UA = "weather-ml-v2/1.0 (https://github.com/yardenmorad2003/weather-ml-v2; ym2705@nyu.edu)"
THUMB_WIDTH = 2560
REQUEST_DELAY_S = 0.3
TIMEOUT_S = 20

COUNTRY_NAMES = {
    "US": "United States", "CA": "Canada", "MX": "Mexico", "CU": "Cuba",
    "PA": "Panama", "CR": "Costa Rica", "JM": "Jamaica", "SV": "El Salvador",
    "GT": "Guatemala", "HN": "Honduras", "DO": "Dominican Republic",
    "PR": "Puerto Rico", "BS": "Bahamas", "HT": "Haiti",
    "AR": "Argentina", "BR": "Brazil", "PE": "Peru", "CO": "Colombia",
    "CL": "Chile", "EC": "Ecuador", "UY": "Uruguay", "BO": "Bolivia",
    "VE": "Venezuela", "PY": "Paraguay",
    "GB": "United Kingdom", "FR": "France", "DE": "Germany", "RU": "Russia",
    "ES": "Spain", "IT": "Italy", "SE": "Sweden", "GR": "Greece",
    "IS": "Iceland", "TR": "Turkey", "PT": "Portugal", "IE": "Ireland",
    "PL": "Poland", "AT": "Austria", "NO": "Norway", "FI": "Finland",
    "RO": "Romania", "UA": "Ukraine", "CH": "Switzerland", "NL": "Netherlands",
    "CZ": "Czech Republic", "HU": "Hungary", "DK": "Denmark", "BE": "Belgium",
    "MT": "Malta", "EE": "Estonia", "LV": "Latvia", "BY": "Belarus",
    "BG": "Bulgaria", "RS": "Serbia", "HR": "Croatia", "BA": "Bosnia and Herzegovina",
    "EG": "Egypt", "NG": "Nigeria", "KE": "Kenya", "ZA": "South Africa",
    "MA": "Morocco", "ET": "Ethiopia", "GH": "Ghana", "TZ": "Tanzania",
    "TN": "Tunisia", "DZ": "Algeria", "RW": "Rwanda", "SN": "Senegal",
    "SD": "Sudan", "CD": "Democratic Republic of the Congo", "AO": "Angola",
    "NA": "Namibia", "ZW": "Zimbabwe", "ZM": "Zambia", "MZ": "Mozambique",
    "CI": "Ivory Coast", "CM": "Cameroon", "MG": "Madagascar", "SO": "Somalia",
    "UG": "Uganda", "ML": "Mali", "BF": "Burkina Faso", "MU": "Mauritius",
    "AE": "United Arab Emirates", "IR": "Iran", "SA": "Saudi Arabia",
    "IL": "Israel", "JO": "Jordan", "QA": "Qatar", "OM": "Oman",
    "LB": "Lebanon", "SY": "Syria", "IQ": "Iraq", "KW": "Kuwait",
    "JP": "Japan", "CN": "China", "IN": "India", "TH": "Thailand",
    "SG": "Singapore", "KR": "South Korea", "MN": "Mongolia", "VN": "Vietnam",
    "NP": "Nepal", "ID": "Indonesia", "HK": "Hong Kong", "TW": "Taiwan",
    "PH": "Philippines", "MY": "Malaysia", "BD": "Bangladesh", "PK": "Pakistan",
    "LK": "Sri Lanka", "GE": "Georgia", "AZ": "Azerbaijan",
    "AU": "Australia", "NZ": "New Zealand", "FJ": "Fiji",
}

BAD_FILENAME_HINTS = ("flag_of", "coat_of_arms", "map_of", "logo", "seal_of", "location_map")


def fetch(title: str) -> dict | None:
    """Query MediaWiki for the lead image. Returns None on miss."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages|pageprops",
        "titles": title,
        "piprop": "original|thumbnail|name",
        "pithumbsize": str(THUMB_WIDTH),
        "redirects": "1",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.load(resp)

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "missing" in page:
            return None
        if "disambiguation" in page.get("pageprops", {}):
            return {"disambiguation": True, "resolved_title": page.get("title")}
        orig = page.get("original")
        thumb = page.get("thumbnail")
        if not orig:
            continue
        return {
            "resolved_title": page.get("title"),
            "original_url": orig["source"],
            "original_width": orig["width"],
            "original_height": orig["height"],
            "thumb_url": thumb["source"] if thumb else orig["source"],
            "thumb_width": thumb["width"] if thumb else orig["width"],
            "thumb_height": thumb["height"] if thumb else orig["height"],
            "file_name": page.get("pageimage", ""),
        }
    return None


def try_titles(city: dict) -> tuple[str, dict] | tuple[str, None]:
    """Cascade: bare name → 'Name, Country' → 'Name (city)'."""
    name = city["name"]
    country = COUNTRY_NAMES.get(city["country"], "")
    candidates = [name]
    if country:
        candidates.append(f"{name}, {country}")
    candidates.append(f"{name} (city)")

    last_attempt = name
    for title in candidates:
        last_attempt = title
        try:
            result = fetch(title)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {title}: {exc}", file=sys.stderr)
            time.sleep(REQUEST_DELAY_S)
            continue
        if result and not result.get("disambiguation"):
            return title, result
        time.sleep(REQUEST_DELAY_S)
    return last_attempt, None


def flag_image(rec: dict) -> list[str]:
    """Return a list of concerns about this image, empty if it looks fine."""
    issues = []
    fn = rec.get("file_name", "").lower()
    for hint in BAD_FILENAME_HINTS:
        if hint in fn:
            issues.append(f"filename contains '{hint}'")
    if rec.get("original_width", 0) < 1500:
        issues.append(f"low-res original ({rec.get('original_width')}px wide)")
    return issues


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    misses: list[str] = []

    total = len(CITIES)
    for i, city in enumerate(CITIES, 1):
        name = city["name"]
        print(f"[{i:3d}/{total}] {name} ...", flush=True)
        tried_title, rec = try_titles(city)
        if rec is None:
            misses.append(name)
            results[name] = {"tried": tried_title, "error": "no suitable image found"}
        else:
            rec["issues"] = flag_image(rec)
            results[name] = rec
        time.sleep(REQUEST_DELAY_S)

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_JSON} ({len(results)} cities, {len(misses)} misses)")
    if misses:
        print("Misses:", ", ".join(misses))

    write_preview(results)
    print(f"Wrote {OUT_HTML}")


def write_preview(results: dict[str, dict]) -> None:
    """Grid preview. Cities with issues get a red outline and a tag."""
    cards = []
    for name, rec in results.items():
        if "error" in rec:
            cards.append(f"""
<div class="card miss">
  <div class="name">{name}</div>
  <div class="meta">miss — tried: {rec.get('tried', '?')}</div>
</div>""")
            continue
        issues = rec.get("issues") or []
        cls = "card" + (" flagged" if issues else "")
        issue_html = ""
        if issues:
            issue_html = f'<div class="issues">{" · ".join(issues)}</div>'
        cards.append(f"""
<div class="{cls}">
  <img loading="lazy" src="{rec['thumb_url']}" alt="{name}" />
  <div class="name">{name}</div>
  <div class="meta">{rec.get('resolved_title', '?')} — {rec['original_width']}×{rec['original_height']}</div>
  {issue_html}
</div>""")

    total = len(results)
    flagged = sum(1 for r in results.values() if r.get("issues"))
    missed = sum(1 for r in results.values() if "error" in r)

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>city images preview</title>
<style>
body {{ background: #0b0b0b; color: #eee; font: 14px system-ui; margin: 0; padding: 24px; }}
h1 {{ margin: 0 0 4px 0; }}
.summary {{ color: #888; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
.card {{ background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 2px solid transparent; }}
.card.flagged {{ border-color: #cc3333; }}
.card.miss {{ border-color: #cc3333; background: #330f0f; min-height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; }}
.card img {{ width: 100%; height: 200px; object-fit: cover; display: block; }}
.name {{ padding: 8px 12px 4px; font-weight: 600; }}
.meta {{ padding: 0 12px 8px; color: #888; font-size: 12px; }}
.issues {{ padding: 4px 12px 10px; color: #ff7777; font-size: 12px; }}
</style>
<h1>city images preview</h1>
<div class="summary">{total} cities — {missed} misses, {flagged} flagged for manual review</div>
<div class="grid">{''.join(cards)}</div>
"""
    OUT_HTML.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
