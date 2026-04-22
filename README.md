# weather-ml-v2

A natural-language climate recommender. Type a city and a vibe — *"Seattle but sunnier"* — and get a ranked list of real cities that fit, with a plain-English breakdown of why each one was picked and where it trades off.

**Live demo:** [yardenmorad2003.github.io/weather-ml-v2](https://yardenmorad2003.github.io/weather-ml-v2/)

The live backend runs on Render's free tier and spins down after 15 min of idle, so the first request of a session takes 30–60 s. Subsequent queries are ~200–500 ms.

---

## What it does

Three tabs:

- **Query** — type free text, see the top 10 matching cities ranked by climatic similarity, each with a factual breakdown ("Why this matches").
- **Explorer** — rotate all 230 canonical cities in a 3D PCA scatter. Axes are auto-labeled by an LLM from their top loadings (e.g. *"Sunny autumn days"*, *"Mild winter temperatures"*, *"Breezy autumn and winter"*).
- **How it works** — an in-app explainer for non-technical users.

### Examples

| Query | Top result | Notable |
|---|---|---|
| `Seattle but sunnier` | Amman (44.2%) | Intent-match wins, score stays low because Amman ≠ Seattle overall |
| `New York but warmer winters` | Casablanca (46.1%) | Mediterranean with NYC-like everything-else |
| `Tokyo but less humid summers` | Madrid (35.7%) | Continental dry-summer climate |
| `London but warmer winters` | Lima (39.4%) | Marine mild winter analog |
| `like Tokyo` (no modification) | Shanghai (48.2%) | Nearest climatic twin |

---

## How it works

```
┌────────────────────┐   free text   ┌────────────────────┐   {anchor,     ┌───────────────────┐
│  Next.js frontend  ├──────────────►│   FastAPI /        ├────vibes}────► │   OpenAI          │
│  (GitHub Pages)    │               │   nl_parser        │               │   (gpt-4o-mini)   │
└─────────┬──────────┘               └──────────┬─────────┘               └───────────────────┘
          │                                     │
          │                                     │ resolve anchor city
          │                                     ▼
          │                          ┌────────────────────┐   lat/lon      ┌───────────────────┐
          │                          │  city_resolver     ├──────────────► │   Open-Meteo      │
          │                          │  (canon→cache→api) │               │   ERA5 archive    │
          │                          └──────────┬─────────┘               └───────────────────┘
          │                                     │
          │                                     ▼ apply vibe deltas in σ-space
          │                          ┌────────────────────┐
          │                          │  recommender       │
          │                          │  weighted-Euclidean│
          │                          │  rank over 230 cty │
          │                          └──────────┬─────────┘
          │                                     │
          │         ranked results +            ▼
          │◄─────── per-vibe factual ───────────┘
          │         diffs + general
          │         feature diffs
          ▼
    user sees cards w/
    similarity %, map,
    and green/gray dots
    per dimension
```

### The pipeline in 7 steps

1. **Parse the query.** `nl_parser.py` calls OpenAI with a Pydantic schema whose fields are `Literal` types over a closed 15-axis × 3-scope × 3-intensity vocabulary. Constrained decoding guarantees the LLM can only emit valid tokens — no hallucinated `less_humid` when only `drier` and `less_muggy` exist.

2. **Resolve the anchor city.** `city_resolver.py` cascades through exact canonical match → fuzzy canonical match (difflib) → SQLite cache → Open-Meteo geocoding + hourly-history fetch → `make_city_profile` → cache. Unknown cities show up with one-time 10-ish-second cold-fetch latency, then are instant forever.

3. **Build the anchor's fingerprint.** Every city has a 96-dimensional profile: 12 months × 8 weather features (temp, humidity, dewpoint, precip, cloud, pressure, wind, clear-sky fraction) averaged from 2 years of hourly ERA5 data.

4. **Phase-align seasons.** `profile.phase_align` rolls each profile so the coldest-temp month sits at index 0. This makes "cold-season slot" mean the same thing in New York and Sydney, and gracefully handles near-equatorial cities where seasonality is weak.

5. **Apply vibe deltas.** `vibe_table.py` maps each `(axis, scope, intensity)` tuple onto specific `(month, feature)` cells in the StandardScaler'd profile. *"Sunnier, year_round, noticeably"* shifts all 12 clear-sky dims up by 2.5σ and all 12 cloud dims down by 2.5σ.

6. **Rank by weighted Euclidean distance.** `recommender.py` measures each city's distance to the modified-anchor vector. The 24 dimensions the vibe touched get a 6× focus weight — they count more toward the ranking but no longer override magnitudes the way cosine similarity did (see the *design principles* section on why).

7. **Explain.** `reasons.py` builds a factual diff per vibe (e.g. *"Temp (cold season): 22.3°C vs 1.7°C (+20.6°C)"*) plus supporting rows on every other human-readable feature, marking each as matched (close to anchor, or moved in the requested direction) or trade-off (differs by more than half a cross-city std).

### The PCA explorer

On top of the same 230 × 96 matrix, `state.py` also fits `PCA(n_components=3)` after StandardScaler. The Explorer tab visualizes all cities in that 3D space. PC1/PC2/PC3 together capture ~60% of the variance. Each axis is labeled by a single OpenAI call from `_label_axes` — constrained to return a 2–5 word label plus a 1–2 sentence plain-English explanation pointing at a known example city at each end.

---

## Design principles

The whole project is organized around a single principle: **the ranking should be honest, and the honesty should be inspectable.**

Concretely:

- **LLM outputs are always schema-constrained.** Nothing downstream trusts a string that came out of the model. Every axis, scope, intensity is a `Literal`. Every axis label pairs with a structured explanation. No regex, no try/except JSON parsing.

- **Similarity metric swapped from cosine → weighted Euclidean.** Cosine measured angular alignment, which let cities that were "pointing the same direction" dominate rankings even when magnitudes disagreed ("Cairo 92% for Seattle but sunnier" because Cairo is max sunny, even though it's also max hot and max dry). Euclidean asks *"how close is this city to the exact modified target?"* which surfaces intent-match at the top but penalizes extreme divergence with a lower score. *Cairo now ranks #3 at 39.1% — top of the sunnier list but honestly flagged as not-really-Seattle.*

- **Per-query score calibration.** The displayed similarity percentage is `exp(-d / sqrt(sum(weights)))` — normalized so scores are comparable across queries regardless of how many dimensions were focus-weighted.

- **Scores communicate honest trade-offs.**
  - 80–90%+ → very close climatic twin
  - 50–60% → reasonable match
  - 30–40% → best available given constraints, but diverges elsewhere
  - <25% → little in the dataset fits

- **Every rank has a factual breakdown.** The "Why this matches" panel on each city card isn't LLM prose — it's real averages in real units (°C, %, mm/h, km/h). The green/gray dots encode *how close* each feature is to the anchor, not how good the match is in some abstract sense. The +/− direction shows which way the city differs, not whether that's good.

- **Phase-based seasonal alignment** replaces naive `lat >= 0 → NH` assumption. Makes the system work cleanly for Northern, Southern, and equatorial cities without hemisphere-specific code paths.

---

## Tech stack

| Layer | Choice |
|---|---|
| LLM parsing | OpenAI `gpt-4o-mini` with Pydantic structured outputs |
| Numerical | `numpy`, `scikit-learn` (StandardScaler + PCA) |
| API | FastAPI + SQLAlchemy + SQLite |
| Weather data | Open-Meteo ERA5 archive (2023–2024 hourly) |
| Frontend | Next.js 16 (App Router, static export), Tailwind 4, Plotly `scatter3d` |
| Deploy | GitHub Pages (frontend) + Render free tier (backend) |
| CI/CD | GitHub Actions workflow on push to `main` |

---

## Running locally

### Backend

```bash
cd backend
python -m venv venv
./venv/Scripts/activate   # Windows; `source venv/bin/activate` elsewhere
pip install -r requirements.txt
cp .env.example .env      # then edit to set OPENAI_API_KEY
uvicorn backend.main:app --reload
```

Run from the *repo root* (not `backend/`) so Python treats `backend` as a package — the internal imports are relative.

First startup is instant because `backend/cache/profiles_2023-01-01_2024-12-31.npz` is committed (40 KB for 230 cities × 96 dims × 4 bytes). If you want to rebuild from scratch for a different year range, edit `HISTORY_START` / `HISTORY_END` in `climatology.py` and `city_resolver.py`, then delete the npz and hit any endpoint — it'll regenerate via Open-Meteo (~10–30 min depending on chunk cache state).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`. The frontend reads `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`) — point it at a deployed backend by setting it in `.env.local` if you want.

---

## Project structure

```
weather-ml-v2/
├── backend/
│   ├── cache/                  # npz climatology, parquet chunks (gitignored except npz)
│   ├── core/
│   │   ├── config.py           # Pydantic settings, reads .env
│   │   └── weather_ml.py       # sys.path shim for vendored modules
│   ├── db/
│   │   ├── models.py           # Vote, FetchedCity
│   │   └── session.py
│   ├── routers/
│   │   ├── recommend.py        # POST /recommend/text
│   │   ├── pca.py              # GET /pca, POST /pca/project
│   │   └── cities.py           # GET /cities/{name}
│   ├── services/
│   │   ├── nl_parser.py        # OpenAI structured output → ParsedQuery
│   │   ├── city_resolver.py    # anchor → 96-d profile cascade
│   │   ├── climatology.py      # multi-year profile builder, 429-retry
│   │   ├── profile.py          # phase_align (seasonal alignment)
│   │   ├── vibe_table.py       # vibe → σ-space delta mapping
│   │   ├── recommender.py      # weighted-Euclidean rank
│   │   ├── reasons.py          # per-result factual diff explainer
│   │   ├── pca_service.py      # PCA(3), LLM axis labels + explanations
│   │   └── state.py            # lazy shared state cache
│   ├── vendor/weather_ml/      # vendored cities + data + features (self-contained)
│   └── main.py                 # FastAPI app + CORS
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Query tab
│   │   ├── explorer/page.tsx   # 3D PCA
│   │   ├── about/page.tsx      # How it works
│   │   ├── components/CityCard.tsx
│   │   └── layout.tsx          # Nav, aurora, dark theme
│   ├── lib/api.ts              # typed fetch wrappers
│   └── next.config.ts          # output: "export" + basePath for Pages
├── .github/workflows/pages.yml # static build → Pages deploy
└── README.md                   # you are here
```

---

## Roadmap

Ordered by impact-per-effort:

- [ ] **30-year climatology via WorldClim** — replace 2-year ERA5 with 30-year monthly normals for the core features. Biggest accuracy lift; smooths single-year anomalies.
- [ ] **Richer features** — diurnal temperature range (day-night swing) and per-month temperature percentiles (P10/P90). Separates Phoenix from Miami in a way annual means can't.
- [ ] **Geo dimensions** — elevation (free from Open-Meteo), distance to coast, terrain ruggedness. Unlocks vibes like `coastal`, `mountainous`, `higher_elevation`.
- [ ] **Multi-anchor blends** — *"between Barcelona and Miami"* → mean of two profiles as the start vector.
- [ ] **Negative anchors** — *"like Tokyo but not Singapore-humid"* pushes *away* from a reference on specific dims.
- [ ] **Wire up the Votes table** — the schema exists; a pairwise-preference model could tune per-feature weights from actual user behavior.
- [ ] **Restore the dropped 60 cities** — Kabul, Sanaa through San Pedro de Atacama were dropped mid-build due to Open-Meteo rate-limiting. Chunks are cached; a paced rerun would finish in one session.

## Known limits

- Only 230 canonical cities are *ranked*. Unknown anchor cities get geocoded + fetched on demand — fine for famous places, may fail for tiny towns.
- A single vibe per query is the sweet spot. Multiple vibes are accepted but each applies at full strength, so *"warmer winters and drier summers"* may over-rotate.
- Free-tier Render cold-starts add 30–60 s on the first request.
- 2-year window ≠ climate normal. An anomalously hot 2023 nudges every fingerprint slightly.

---

## Credits

- Historical weather: [Open-Meteo](https://open-meteo.com) ERA5 archive (no auth required, thank you).
- Natural-language parsing: OpenAI `gpt-4o-mini` with constrained structured outputs.
- Visualization: [Plotly](https://plotly.com/javascript/) `scatter3d`.
- Originally built on top of a sibling `weather-ml` experimentation repo; the relevant data-fetching and profile-building utilities are vendored under `backend/vendor/weather_ml/` for self-contained deployment.
