# Contrastive deployment — handoff to next session

Companion to `CONTRASTIVE_PLAN.md` (overall plan) and `CONTRASTIVE_DATA.md`
(synthetic dataset state). This doc covers what's trained, where the
artifacts live, the deployment architecture decision, and the next concrete
build steps.

This is for the CSCI-UA 473 final project (presentation Week 14,
Apr 27 – May 1 2026). Repo at `C:\Users\yarde\weather-ml-v2`,
deployed at:
- Frontend: https://yardenmorad2003.github.io/weather-ml-v2/
- Backend: https://weather-ml-v2.onrender.com

## TL;DR

Stage 1 of the contrastive plan is **done and validated**. The dual-encoder
model trains end-to-end and beats the rubric target (recall@10 = 0.804 vs
target 0.70). Trained artifacts live in `backend/cache/` and total <200 KB.

Decision made on 2026-04-26: **Stage 1 inference will run in the browser,
not the backend.** Backend grows by exactly one static endpoint; the
sentence-transformer + ONNX runtime ship to the client and cache in
IndexedDB after first download.

The next session should pick up at "deployment work" below.

## Where things stand

| Stage | Status |
|---|---|
| Synthetic dataset (10k triples) | Complete (10,000 / 10,000 labeled) |
| Stage 1 architecture + losses | Implemented, hand-written InfoNCE + triplet |
| Stage 1 training | Trained on CPU locally; recall@10 = 0.804 |
| Stage 1 evaluation | recall@1/5/10 + per-type breakdown + t-SNE — done |
| **Deploy in the live app** | **Not started — pick up here** |
| Stage 2 (per-user adapter) | Not started |
| DB vote logging (`Vote` table) | Schema exists; nothing writes to it |

## Trained artifacts (`backend/cache/`)

| File | Size | Purpose |
|---|---|---|
| `text_proj_state.pt` | 51 KB | Trained 384→32 projection (single linear) |
| `city_encoder_state.pt` | 36 KB | Trained 96→64→32 city MLP |
| `city_scaler.npz` | 1.3 KB | StandardScaler `mean` and `scale` for input normalization |
| `city_embeddings_learned.npz` | 33 KB | Final 230×32 L2-normalized cache for cosine ranking |
| `contrastive_test_split.jsonl` | 104 KB | 1,000 held-out queries for reproducible eval |
| `contrastive_tsne.png` | 95 KB | Slide-ready visualization of city embeddings |
| `contrastive_tsne_coords.json` | 33 KB | (230, 2) coords + continent labels for frontend rendering |

Total weight footprint: **~120 KB**. Small enough to commit or to inline
into a frontend manifest. No LFS needed.

## Headline metrics (held-out 1k split)

| Metric | Score |
|---|---|
| recall@1 | 0.284 |
| recall@5 | 0.672 |
| **recall@10** | **0.804** (target >0.70) |

Per query type:

| Type | Count | Recall@10 |
|---|---|---|
| Descriptive ("warm coastal sunny") | 355 | **0.890** |
| Numerical ("winter highs above 50F") | 43 | 0.837 |
| Anchored ("X but Y") | 562 | 0.758 |
| Multi-anchor ("between X and Y") | 40 | 0.650 |

Demo-grade insight from the t-SNE: the model clusters cities **by climate
type, not by continent**. Cape Town ↔ Sydney, Reykjavik ↔ Anchorage,
Cairo + Dubai + Phoenix all sit together. Three continents collapse into
one "hot desert" point. This is the most important slide.

## Code map

```
backend/
  services/
    contrastive.py              # CityEncoder, TextProjection,
                                # hand-written InfoNCE + triplet loss
  scripts/
    precompute_embeddings.py    # frozen MiniLM forward over 10k queries
    train_contrastive.py        # full train loop + eval folded in
    qualitative_eval.py         # probe queries + per-type recall + t-SNE
    generate_triples.py         # (already done) synthetic dataset gen
notebooks/
  train_colab.ipynb             # thin runner for Colab T4
```

All run as `python -m backend.scripts.<name>` from repo root.

## Architectural decision: browser-side inference

The plan doc originally assumed backend inference (a `/contrastive/recommend`
endpoint loading torch + transformers + MiniLM at startup). After measuring
the actual model size and weighing the deployment context, **we're going
browser-side instead**.

### Why

1. **Render free-tier image budget.** Adding torch + transformers + MiniLM
   weights to the backend would consume ~400 MB and likely break the 512 MB
   limit. Browser-side keeps the backend exactly the size it is today.
2. **Stage 2 fits naturally.** The per-user adapter is 1056 params and trains
   in <1 s on a phone CPU. No `/finetune` endpoint or server-stored adapter
   weights — adapter lives in `localStorage` keyed by session ID.
3. **Cold starts.** Render free-tier sleeps. Browser-side inference removes
   the contrastive feature from the cold-start dependency chain — once
   MiniLM is cached, queries don't hit the backend at all.
4. **Repo pattern fit.** Frontend is already `output: "export"` (static
   site, client components only). Browser ML is on-pattern.

### What it costs

- One-time download of **~18 MB** (quantized MiniLM ONNX + WASM runtime
  + tokenizer) on the user's first contrastive query. Loaded lazily, cached
  in IndexedDB. Subsequent visits on the same browser are instant.
- Mitigated by **background preload** on page mount: the model fetches in
  the background while the user is reading the homepage, hiding most of
  the cost.
- Per-device caching: a user who visits on their phone *and* laptop pays
  once on each. Incognito wipes the cache.

### What changes on the backend

Exactly one new endpoint:

```
GET /contrastive/manifest -> {
  cities: [...],                          # 230 names, ordered
  embeddings: <(230,32) float32 array>,   # cosine-ready, L2-normed
  projection: {weight: [[...]], bias: [...]},  # 384→32 trained head
  scaler: {mean: [...], scale: [...]},    # for safety; mostly unused if backend never re-encodes
}
```

That's it. **No torch import in the backend.** The whole manifest is
~50 KB serialized JSON; cached forever via standard HTTP cache headers.

### What changes on the frontend

New module `frontend/lib/contrastive.ts`:
- Lazy-loads `Xenova/all-MiniLM-L6-v2` (quantized, ~13 MB).
- Fetches the manifest once and caches in memory.
- `rankContrastive(query: string, topK = 10) -> CityResult[]`:
  encode → linear projection (6-line matmul, no torch in browser) →
  cosine vs cached embeddings → top-K.

## Open work, in priority order

| Step | Time | What |
|---|---|---|
| 1. Backend `/contrastive/manifest` endpoint | ~30 min | Pure static JSON serializer. No new dependencies. |
| 2. Frontend `lib/contrastive.ts` (browser-side ranker) | ~2 h | `@xenova/transformers` lazy-load + projection matmul + cosine ranking |
| 3. UX integration (Smart toggle on Query tab) | ~1 h | Same input, swap ranker via toggle. Default to classical. |
| 4. Side-by-side demo view (`?compare=1`) | ~30 min | Hidden URL flag → split screen, both rankers on same query. Slide-grade. |
| 5. DB vote logging | ~1 h | Wire the existing `Vote` table; write on `/tournament/final` |
| 6. Tournament 20-round "Personalized" toggle | ~1 h | Default stays at 10 rounds; toggle extends to 20 |
| 7. Per-user adapter (Stage 2) in browser | ~3 h | 1056-param `W_user · f(city) + b`, hand-written triplet loss, train on 20 picks, store in `localStorage` |
| 8. Demo polish: arch diagram, before/after viz, slides | ~2 h | Slide assets — uses outputs from steps 4 + 7 |

**MVP path** = steps 1-3 (~3.5 h). After that, Smart toggle works for any user.
**Demo-complete path** = + steps 4 + 7 + 8 (~6 h). Personalization story.
**Full path** = all 8 (~11 h).

Items 5 + 6 are independent of 1-4 and can be parallelized to a different
team member.

## Files to read first in the next session

In order:
1. `CONTRASTIVE_PLAN.md` — overall plan, esp. §"Module structure" and
   §"Things to NOT do".
2. **This file (`CONTRASTIVE_DEPLOY.md`)** — current state + next steps.
3. `backend/services/contrastive.py` — the model.
4. `backend/services/recommender.py`, `backend/services/tournament.py`,
   `backend/routers/tournament.py` — existing patterns; new code should
   match these.
5. `frontend/app/tournament/page.tsx` and `frontend/lib/api.ts` — existing
   frontend patterns; new code should match these.

## Running things

To re-train end-to-end (only if needed; existing artifacts are fine):

```bash
# 1. Precompute MiniLM embeddings (one-time, ~2 min on CPU)
./backend/venv/Scripts/python -u -m backend.scripts.precompute_embeddings \
    --in backend/cache/contrastive_triples.jsonl \
    --out backend/cache/text_embeddings.npz

# 2. Train (~2-5 min on CPU, ~30 sec on Colab T4)
./backend/venv/Scripts/python -u -m backend.scripts.train_contrastive \
    --triples backend/cache/contrastive_triples.jsonl \
    --text-emb backend/cache/text_embeddings.npz \
    --profiles backend/cache/profiles_2023-01-01_2024-12-31.npz \
    --out-dir backend/cache \
    --epochs 100 --batch-size 256 --lr 1e-3 --seed 42 --eval-every 10

# 3. Qualitative eval (probes + per-type recall + t-SNE)
./backend/venv/Scripts/python -u -m backend.scripts.qualitative_eval \
    --out-dir backend/cache
```

For Colab T4 (faster + less local CPU pressure):
open `notebooks/train_colab.ipynb`, follow the cells. Upload the two cache
files when prompted; download the artifact zip at the end.

## Things to NOT do

- Don't add `torch` or `transformers` to backend `requirements.txt`. The
  whole point of the deployment decision is to keep the backend slim.
- Don't fetch the MiniLM model eagerly on every page; it's lazy + background-
  preloaded.
- Don't break the Query, Explorer, or Tournament tabs. Contrastive is
  additive (toggle / new tab / new mode).
- Don't commit `text_embeddings.npz` (it's 16 MB and gitignored). Stick to
  the trained `.pt` + small `.npz` outputs.
- Don't re-run the labeling step. The 10k triples in
  `backend/cache/contrastive_triples.jsonl` are gitignored locally — back
  up before any cleanup.
- Don't fake the demo. If a query returns weird results, show it as-is and
  use it to motivate the per-user adapter slide.
- Don't blend the contrastive and classical rankers into a single fused
  score. The demo story is two clearly separable rankers, side-by-side.

## Why this matters for the rubric

- **Algorithm Implementation (10%)** — covered. Symmetric InfoNCE and
  triplet loss are both hand-written in `backend/services/contrastive.py`
  with explicit log-softmax, no `F.cross_entropy` shortcut.
- **Working Demo (10%)** — pending steps 1-3 above. Trained weights are
  tiny enough to commit; no per-grader retraining needed.
- **Live Presentation (10%)** — t-SNE + per-type recall + side-by-side
  comparison view give every team member something concrete to walk
  through.
- **PCA caveat** — being handled separately (another teammate).
