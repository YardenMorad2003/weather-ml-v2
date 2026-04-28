# weather-ml-v2

A natural-language climate recommender. Type a city and a vibe — *"Seattle but sunnier"* — and get a ranked list of real cities that fit, with a plain-English breakdown of why each one was picked and where it trades off.

The system has **two complementary rankers** and routes between them by query shape:

- **Classical** — anchor extraction (gpt-4o-mini, schema-constrained) → vibe deltas in σ-space → weighted Euclidean over 230 cities. Strong on *"X but Y"* queries.
- **Contrastive (Stage 1)** — dual-encoder (frozen MiniLM 384-d → trained 32-d head; 96-d city profile → trained 32-d head) trained from scratch with hand-written InfoNCE + triplet loss on 10k synthetic LLM-labeled triples. Strong on free-form vibe queries (*"warm coastal sunny"*). Runs entirely **in the browser** — backend only serves a 405 KB static manifest.

**Live demo:** [yardenmorad2003.github.io/weather-ml-v2](https://yardenmorad2003.github.io/weather-ml-v2/) · side-by-side: [`?compare=1`](https://yardenmorad2003.github.io/weather-ml-v2/?compare=1)

The live backend runs on Render's free tier and spins down after 15 min of idle, so the first request of a session takes 30–60 s. Subsequent queries are ~200–500 ms. Once the contrastive model is cached in IndexedDB (one-time ~13 MB download from HuggingFace on first Smart query), Smart-mode rankings happen entirely client-side with no backend round-trip.

---

## What it does

Four tabs:

- **Query** — type free text, see the top 10 matching cities, each with a factual breakdown ("Why this matches"). A three-way ranker toggle (Auto / Classical / Smart) lets you pick the recommender. **Auto** is the default and routes on what the parser extracted: anchor *or* any structured vibe → classical (the σ-eval shows it wins decisively on vocabulary-covered queries); empty parser output (no anchor, no vibes) → contrastive (queries the vibe vocabulary can't represent). A `?compare=1` URL flag flips into a side-by-side mode that runs both rankers on the same query and highlights cities that appear in both top-10 lists.
- **Explorer** — rotate all 230 canonical cities in a 3D PCA scatter. Axes are auto-labeled by an LLM from their top loadings (e.g. *"Sunny autumn days"*, *"Mild winter temperatures"*, *"Breezy autumn and winter"*). The PCA itself is a from-scratch implementation (covariance → `eigh` → top-k components) — sklearn's `PCA` is no longer a runtime dependency for this tab.
- **Tournament** — pairwise picks across 10 rounds; the final round shows the implied climate preferences ranked over all cities. Designed as the data source for the planned per-user adapter (Stage 2).
- **How it works** — an in-app explainer for non-technical users.

### Examples

Classical ranker — strong when the query has an anchor and a modifier:

| Query | Top result | Notable |
|---|---|---|
| `Seattle but sunnier` | Amman (44.2%) | Intent-match wins, score stays low because Amman ≠ Seattle overall |
| `New York but warmer winters` | Casablanca (46.1%) | Mediterranean with NYC-like everything-else |
| `Tokyo but less humid summers` | Madrid (35.7%) | Continental dry-summer climate |
| `London but warmer winters` | Lima (39.4%) | Marine mild winter analog |
| `like Tokyo` (no modification) | Shanghai (48.2%) | Nearest climatic twin |

Contrastive ranker — strong when the query is free-form vibe text with no anchor. The model also exposes its own confidence: a wide spread between the #1 and #10 score (≥ 15 pts) is shown as **confident**; a narrow spread (< 8 pts) is flagged as **uncertain** so the user knows to discount the ranking.

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

### The contrastive ranker (Stage 1)

A dual-encoder maps free-text queries and city weather profiles into a shared 32-d space, then ranks by cosine similarity. The architecture is small (~20K trainable params) and trained from scratch on synthetic data:

- **Text side:** frozen `sentence-transformers/all-MiniLM-L6-v2` (22M, no fine-tuning) → trainable single linear `384 → 32`.
- **City side:** trainable MLP `96 → 64 → 32` (ReLU between) over the same StandardScaler-normalized profile vector the classical recommender uses.
- **Loss:** symmetric InfoNCE with in-batch negatives (CLIP-style), hand-written from log-softmax + diagonal indexing — no `F.cross_entropy` shortcut, so the algorithm-from-scratch claim is end-to-end. Triplet loss is also implemented from scratch in the same module for use by Stage 2 (per-user adapter, planned).
- **Training data:** 10k synthetic `(query, positives, negatives)` triples generated by gpt-4o-mini with constrained-decoding over the canonical 230-city list. Training takes <2 min on a Colab T4 once MiniLM embeddings are precomputed.
- **Eval:** held-out 1k split, **recall@10 = 0.804** (target was > 0.70). Per query type: descriptive 0.890, numerical 0.837, anchored 0.758, multi-anchor 0.650.

**Inference is browser-side.** The backend serves `GET /contrastive/manifest` — a single 405 KB JSON containing the 230 × 32 city embeddings, the trained `32 × 384` projection head, the StandardScaler stats, and dim metadata. `frontend/lib/contrastive.ts` lazy-loads MiniLM (q8 ONNX, ~13 MB) via `@huggingface/transformers`, runs the projection matmul + L2 normalize + cosine in plain TypeScript, and caches both the model and the manifest in IndexedDB. After the first query, every subsequent query stays entirely in the browser — no `torch` or `transformers` in `backend/requirements.txt`. See `CONTRASTIVE_DEPLOY.md` for the full deployment rationale.

The training pipeline lives in `backend/scripts/`:

- `generate_triples.py` — LLM-synthetic dataset generation
- `precompute_embeddings.py` — frozen MiniLM forward over the 10k queries (~2 min CPU)
- `train_contrastive.py` — full training loop, dual-encoder + InfoNCE
- `qualitative_eval.py` — recall@k breakdown + t-SNE of the learned space
- `build_contrastive_manifest.py` — converts the trained `.pt` + `.npz` artifacts into the static JSON the API serves (so the runtime doesn't need torch)

A demo-ready t-SNE of the city embeddings shows the model clustering by **climate type, not continent** — Cape Town ↔ Sydney, Reykjavik ↔ Anchorage, Cairo + Dubai + Phoenix collapse into one "hot desert" point.

### Routing between rankers

Auto mode runs the classical request first (it's fast and parses the query as a side-effect). If the parser produced an anchor *or* any vibe, classical handles the ranking. Otherwise — when the vibe vocabulary can't represent the intent at all — Auto calls the contrastive ranker. The σ-space evaluation (see *Evaluation* below) shows classical wins by ≈+0.65 mean cosine across 3,150 synthetic queries against the structured-vibe interpretation of the query, decisive evidence for this rule. Contrastive's role is the residual: queries the parser produces no structured output for. The user can manually override via the toggle, and `?compare=1` bypasses routing and runs both for side-by-side comparison.

### Honest error states

The recommender refuses to return misleading results in three specific cases — each surfaced as a typed response field rather than a fake ranking:

- **Unresolvable anchor (`AnchorError`).** `"Hogwarts but sunny"` → the parser catches "Hogwarts" but the resolver can't place it. Returns did-you-mean suggestions instead of falling through to a centroid-based ranking that would honest-look like an answer (the bug this guard was added to prevent: `"Hogwarts but sunny"` → Cairo at 47%).
- **Contradictory vibes (`VibeConflictError`).** `"drier and more humid"` contradicts on humidity. The conflict guard derives directly from `AXIS_FEATURES` — feature-overlap with opposing signs over shared months — so it stays in sync with vibe-table calibration. Fires on `drier+more_humid`, `wetter+sunnier`, hard direct opposites, and the special-branch `milder+more_extreme` / `more_seasonal+less_seasonal` pairs.
- **Anchor at axis extreme (`SaturationError`).** `"Singapore but more humid"` → Singapore is the apex of humidity in the canonical 230, so the closest neighbors are slightly *less* humid than Singapore itself. The guard fires when `|observed_delta| / |expected_delta| < 0.20` and σ-cosine `< 0` in the touched-dim subspace — i.e., the achievable top-K moves the wrong way and barely at all.

Together the three guards filter ≈5% of queries upfront, dropping the directional-failure rate on what the user actually sees from 5.3% (single-vibe) and 4.1% (compound) baselines down to 0.83% / 1.58% post-guard.

### Evaluation

Five σ-space eval scripts live in `backend/scripts/`. They construct synthetic queries from `(anchor, axis, scope, intensity)` tuples (no parser, no LLM in the loop), apply the vibe table to compute an *expected delta* in StandardScaler-normalized profile space, and score each ranker's top-K against it via cosine similarity in the touched-dim subspace. Closed-form ground truth, no LLM judge, no humans.

- `sigma_eval.py` — anchored 1-vibe (600 queries). The vibe-table regression harness — tweak a coefficient, re-run, see if cosine moved the right way.
- `sigma_eval_compound.py` — anchored 2-vibe (2,400 queries) across 40 compatible axis pairs. Surfaces compound-specific failures.
- `sigma_eval_full.py` — comprehensive 4-mode head-to-head (anchored × free-form, 1-vibe × 2-vibe), classical vs contrastive on the same structured ground truth (3,150 queries total).
- `sigma_eval_freeform.py` — focused free-form classical-vs-contrastive comparison.
- `sigma_eval_inspect.py` — qualitative case-by-case inspection with full per-feature delta breakdown.

Headline (post-guard, mean σ-cosine):

| mode | classical | contrastive | classical's lead |
|---|---|---|---|
| anchored 1-vibe | +0.900 | +0.205 | +0.694 |
| anchored 2-vibe | +0.797 | +0.134 | +0.663 |
| free-form 1-vibe | +0.991 | +0.474 | +0.517 |
| free-form 2-vibe | +0.941 | +0.395 | +0.546 |

The σ-cosine metric is **structurally biased toward classical** — classical literally optimizes the σ-space delta the metric measures. The interpretation isn't "classical is 4× better in human judgment"; it's "classical is decisively better at moving along the exact dimensions the vibe table specifies." For queries where the vocabulary is lossy (`alpine`, `monsoon`, `Hogwarts`), this metric returns a meaningless null because there are no touched dims to score. That's the residual the router puts into contrastive's bucket.

Per-axis breakdown (single-vibe, after vibe-table recalibration):

```
windier        +0.936    less_muggy     +0.919
calmer         +0.915    colder         +0.889
drier          +0.854    cloudier       +0.813
sunnier        +0.807    wetter         +0.783
warmer         +0.732    more_humid     +0.704
```

Recalibrating `warmer`'s dewpoint coupling (+0.5σ → +0.8σ to match the empirical inter-feature correlation +0.80) and dropping `more_humid`'s dewpoint coupling entirely (the data shows hum/dew correlation is only +0.35, not the asserted +0.5) lifted those two worst axes from +0.679 → +0.732 and +0.623 → +0.704 respectively. The σ-eval is the regression harness for any future calibration tweaks — every change runs through it before merging.

### The PCA explorer

On top of the same 230 × 96 matrix, `state.py` also fits `PCA(n_components=3)` after StandardScaler. The Explorer tab visualizes all cities in that 3D space. PC1/PC2/PC3 together capture ~60% of the variance. Each axis is labeled by a single OpenAI call from `_label_axes` — constrained to return a 2–5 word label plus a 1–2 sentence plain-English explanation pointing at a known example city at each end.

The PCA itself is a from-scratch implementation in `state.py`: covariance matrix → `np.linalg.eigh` (real-symmetric guarantee) → eigenvectors as columns of `components_`, sorted by descending eigenvalue. Verified to match sklearn's `PCA` on explained-variance ratios to 4 decimals and per-axis projections to perfect correlation (sign flips in eigenvectors are expected and don't affect visualization).

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
| Numerical | `numpy`, `scikit-learn` (StandardScaler only at runtime; PCA is from-scratch) |
| Contrastive training | PyTorch (training only — not a runtime dep), `sentence-transformers` for the frozen MiniLM forward |
| Browser ML | `@huggingface/transformers` (ONNX runtime + WASM, q8 quantized MiniLM ~13 MB) |
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
│   │   ├── cities.py           # GET /cities/{name}
│   │   ├── tournament.py       # POST /tournament/pair, /tournament/final
│   │   └── contrastive.py      # GET /contrastive/manifest (static)
│   ├── services/
│   │   ├── nl_parser.py        # OpenAI structured output → ParsedQuery
│   │   ├── city_resolver.py    # anchor → 96-d profile cascade
│   │   ├── climatology.py      # multi-year profile builder, 429-retry
│   │   ├── profile.py          # phase_align (seasonal alignment)
│   │   ├── vibe_table.py       # vibe → σ-space delta mapping
│   │   ├── recommender.py      # weighted-Euclidean rank
│   │   ├── reasons.py          # per-result factual diff explainer
│   │   ├── pca_service.py      # LLM axis labels + explanations
│   │   ├── state.py            # lazy shared state cache + from-scratch PCA
│   │   ├── contrastive.py      # dual-encoder + hand-written InfoNCE/triplet (training only)
│   │   └── contrastive_manifest.py  # JSON manifest loader (no torch dep)
│   ├── scripts/
│   │   ├── generate_triples.py        # LLM-synthetic dataset gen
│   │   ├── precompute_embeddings.py   # frozen MiniLM forward over 10k queries
│   │   ├── train_contrastive.py       # full training loop
│   │   ├── qualitative_eval.py        # recall@k + t-SNE
│   │   ├── build_contrastive_manifest.py  # .pt/.npz → JSON for the API
│   │   ├── sigma_eval.py              # 1-vibe σ-space eval (vibe-table regression harness)
│   │   ├── sigma_eval_compound.py     # 2-vibe σ-space eval across 40 axis pairs
│   │   ├── sigma_eval_full.py         # comprehensive 4-mode classical-vs-contrastive
│   │   ├── sigma_eval_freeform.py     # focused free-form comparison
│   │   └── sigma_eval_inspect.py      # qualitative case-by-case σ-eval inspection
│   ├── cache/
│   │   └── contrastive_manifest.json  # 405 KB committed runtime artifact
│   ├── vendor/weather_ml/      # vendored cities + data + features (self-contained)
│   └── main.py                 # FastAPI app + CORS
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Query tab (Auto/Classical/Smart toggle, ?compare=1)
│   │   ├── explorer/page.tsx   # 3D PCA
│   │   ├── tournament/page.tsx # pairwise picks
│   │   ├── about/page.tsx      # How it works
│   │   ├── components/CityCard.tsx
│   │   └── layout.tsx          # Nav, aurora, dark theme
│   ├── lib/
│   │   ├── api.ts              # typed fetch wrappers (classical + tournament)
│   │   └── contrastive.ts      # browser ranker: MiniLM + projection + cosine
│   └── next.config.ts          # output: "export" + basePath for Pages
├── notebooks/
│   └── train_colab.ipynb       # Colab T4 runner for train_contrastive
├── .github/workflows/pages.yml # static build → Pages deploy
├── CONTRASTIVE_PLAN.md         # Stage 1 + 2 plan, architecture, training
├── CONTRASTIVE_DATA.md         # synthetic dataset state + LLM prompts
├── CONTRASTIVE_DEPLOY.md       # browser-vs-backend decision + step-by-step
└── README.md                   # you are here
```

---

## Roadmap

Ordered by impact-per-effort:

- [ ] **Stage 2 contrastive (per-user adapter)** — freeze both encoders, learn a 32×32 user-specific linear adapter from 20-round tournament picks via triplet loss (~1 sec on phone CPU). Lives in `localStorage`, no backend storage. The "personalization" demo slide. Designed and specced; not yet implemented. See `CONTRASTIVE_PLAN.md` § Stage 2.
- [ ] **Improve contrastive on anchor queries** — current model systematically ignores *"but X"* modifiers because synthetic training triples taught it "queries with X → cities like X." A targeted regeneration where the LLM is explicitly instructed to make positives *differ* from the anchor on the requested dimension might salvage modifier handling.
- [ ] **Wire up the Votes table** — schema exists; tournament picks aren't logged yet. Needed to feed Stage 2 in production (today the 20 picks live only in browser state for that session).
- [ ] **30-year climatology via WorldClim** — replace 2-year ERA5 with 30-year monthly normals for the core features. Biggest accuracy lift; smooths single-year anomalies.
- [ ] **Richer features** — diurnal temperature range (day-night swing) and per-month temperature percentiles (P10/P90). Separates Phoenix from Miami in a way annual means can't.
- [ ] **Geo dimensions** — elevation (free from Open-Meteo), distance to coast, terrain ruggedness. Unlocks vibes like `coastal`, `mountainous`, `higher_elevation`.
- [ ] **Multi-anchor blends** — *"between Barcelona and Miami"* → mean of two profiles as the start vector.
- [ ] **Negative anchors** — *"like Tokyo but not Singapore-humid"* pushes *away* from a reference on specific dims.
- [ ] **Restore the dropped 60 cities** — Kabul, Sanaa through San Pedro de Atacama were dropped mid-build due to Open-Meteo rate-limiting. Chunks are cached; a paced rerun would finish in one session.

## Known limits

- Only 230 canonical cities are *ranked*. Unknown anchor cities get geocoded + fetched on demand — fine for famous places, may fail for tiny towns.
- Compound queries (2+ vibes) score ≈0.10 lower in σ-cosine than single-vibe — the cost of asking for two things at once. Direction stays right, magnitudes shrink. Truly contradictory pairs (`drier+more_humid`, `wetter+sunnier`) are filtered upfront by the conflict guard.
- The contrastive ranker treats *"X but Y"* like *"X"* — it learned climate clusters, not anchor-and-tweak. The σ-eval shows it scoring +0.20 mean cosine on anchored 1-vibe queries vs classical's +0.90, so Auto routes any vocabulary-covered query to classical and keeps contrastive for the residual. Use `?compare=1` to see the divergence directly.
- Free-tier Render cold-starts add 30–60 s on the first request.
- First Smart-mode query of a session downloads ~13 MB from HuggingFace's CDN (cached forever in IndexedDB after that, even across page reloads). Incognito wipes the cache.
- 2-year window ≠ climate normal. An anomalously hot 2023 nudges every fingerprint slightly.

---

## Credits

- Historical weather: [Open-Meteo](https://open-meteo.com) ERA5 archive (no auth required, thank you).
- Natural-language parsing: OpenAI `gpt-4o-mini` with constrained structured outputs.
- Visualization: [Plotly](https://plotly.com/javascript/) `scatter3d`.
- Originally built on top of a sibling `weather-ml` experimentation repo; the relevant data-fetching and profile-building utilities are vendored under `backend/vendor/weather_ml/` for self-contained deployment.
