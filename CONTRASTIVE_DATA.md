# Contrastive training dataset — status and handoff

Companion to `CONTRASTIVE_PLAN.md`. This doc covers the synthetic dataset
generation step (item 1-2 in the plan's "Suggested order of work") — what's
built, what's still in flight, and exactly how to resume.

## Where things stand

| Stage | Status | File | Rows |
|---|---|---|---|
| Query generation | **Complete** | `backend/cache/queries.jsonl` | 10,000 |
| Triple labeling  | **In progress** — blocked on OpenAI RPD | `backend/cache/contrastive_triples.jsonl` | 1,973 / 10,000 |

Everything is resumable. The script appends to JSONL and dedupes on load
by normalized query string, so re-running the same command picks up
where it left off.

## The blocker

The OpenAI account is on **Tier 1**, which caps gpt-4o-mini at
**10,000 requests per day** under a **rolling 24-hour window** (not a
midnight UTC reset — see [OpenAI rate-limit docs](https://platform.openai.com/docs/guides/rate-limits)).

Yesterday's labeling attempts burned through the daily quota in two
bursts (roughly 10:30 AM EDT and 11:30 AM-12:00 PM EDT). Quota recovers
gradually as those requests roll off — the cleanest moment to resume is
**after 12:00 PM EDT today** (~24h after the heaviest burst), when the
bulk of slots are free again.

Failed `429` responses also count against the daily limit, so do **not**
restart at high concurrency hoping retries will succeed — they will just
re-burn quota for nothing.

## Resume — one command

After ~12:00 PM EDT, from the repo root:

```bash
./backend/venv/Scripts/python -u -m backend.scripts.generate_triples label \
    --in backend/cache/queries.jsonl \
    --out backend/cache/contrastive_triples.jsonl \
    --concurrency 4
```

Expected behavior:
- Skips the 1,973 rows already done.
- Processes ~8,000 remaining queries in batches of 5 (`LABEL_BATCH_SIZE`),
  so ~1,600 API calls total.
- ~10 minutes wall time at concurrency 4 if quota holds.
- Prints a milestone every 1,000 queries: `labeled 1000 / 8000  (+125/125 this wave, total succeeded 1004)`.

If you start seeing failure rates above ~10% per wave, kill it (`Ctrl-C`),
wait an hour, and resume — it means quota is still recovering.

## What's been built

### `backend/scripts/generate_triples.py`

One script, three subcommands:

| Subcommand | Purpose |
|---|---|
| `queries`  | Generate diverse free-text queries via OpenAI. Resumable, dedup. |
| `label`    | For each query, ask OpenAI to pick 3 positives + 5 negatives from the canonical 230-city list. Resumable. Constrained via Pydantic `Literal[230 cities]` so the model cannot return off-list names. |
| `review`   | Print N random samples from any JSONL (queries or triples) for manual spot-check. |

CLI is `python -m backend.scripts.generate_triples <subcommand> --help`.

### Data files

**`backend/cache/queries.jsonl`** — one query per line:
```json
{"query": "like Bangkok but with cooler nights"}
```

**`backend/cache/contrastive_triples.jsonl`** — one triple per line:
```json
{"query": "like Bangkok but with cooler nights", "positives": ["Mexico City", "Bogota", "Quito"], "negatives": ["Phoenix", "Cairo", "Murmansk", "Reykjavik", "Anchorage"]}
```

Both are gitignored (cache directory rule); they will not be committed.

## Key design decisions worth knowing

1. **Query mix.** The 10k queries cover five styles (anchored / descriptive / preference / multi-anchor / loose comparison) with ~3% containing concrete numerical thresholds (`"under 600mm of rain"`, `"winter highs above 60 degrees"`). Out-of-canon anchor cities (Chamonix, Aspen, etc.) appear intentionally — they exercise exactly the cases where the contrastive model should outperform the classical recommender.

2. **Forbidden modifiers.** The query-generation prompt explicitly bans non-climate modifiers (cost, walkability, food, accessibility, density, scenery, dates). After tightening, the off-topic drift rate is effectively 0% in 50-sample spot-checks.

3. **Batched labeling.** Each label call sends 5 queries at once with a single copy of the candidate-city list, cutting per-query token cost ~4x. The model echoes each query string in its response so we match results to inputs by string (robust to length mismatches). `LABEL_BATCH_SIZE = 5` was validated on the first 1,973 rows; larger batches were considered but rejected as untested under the current prompt.

4. **Concurrency.** Set to 4 for labeling. The math: 4 calls × 1.5s × 1100 tokens/call = ~176k TPM, under the Tier 1 200k cap. Going higher trips TPM bursts; going lower wastes time.

5. **Resumability.** All long jobs use append-only JSONL plus a load-and-skip-on-restart pass. A fresh `queries` or `label` run reads the existing output, builds a `done_set` of normalized query strings, and only attempts what's missing. Safe to interrupt any time.

## Quality controls before training

Before feeding `contrastive_triples.jsonl` into the InfoNCE training loop
(plan item 3), spot-check 100 random triples:

```bash
./backend/venv/Scripts/python -u -m backend.scripts.generate_triples review \
    --in backend/cache/contrastive_triples.jsonl --n 100 --seed 42
```

Look for:
- **Obvious wrong positives** — e.g. "tropical city" with Reykjavik in positives. ~5-10% noise is fine; >20% means re-run labeling with a tightened prompt.
- **Lazy outputs** — same 5-10 cities dominating positives across very different queries (suggests the model defaulted instead of reasoning).
- **Off-topic anchors** — query references a city the model clearly doesn't know well (negatives look random).

If the noise rate looks <10%, ship to training. If higher, the cleanest fix
is to identify the failure mode, edit `LABEL_SYSTEM_PROMPT` in
`generate_triples.py`, delete the bad slice from
`contrastive_triples.jsonl`, and re-run `label`.

## What's next after labeling completes

In order, per the plan:

1. **InfoNCE training loop** in PyTorch (`backend/scripts/train_contrastive.py`)
   on a Colab T4 — text encoder is frozen MiniLM + trainable projection head,
   city encoder is a small trainable MLP. ~20K trainable params. Target
   recall@10 > 70% on a 10% held-out test set.
2. **Per-user adapter** — 1056-param `W_user · f(city) + b` trained via triplet
   loss on a 20-round tournament's picks. <1s on CPU.
3. **Backend endpoints** at `/contrastive/recommend` and `/contrastive/finetune`.
4. **Frontend toggle** for the new ranker (additive — do not break existing tabs).

See `CONTRASTIVE_PLAN.md` §"Module structure (proposed)" and §"Suggested
order of work" for full sequencing.
