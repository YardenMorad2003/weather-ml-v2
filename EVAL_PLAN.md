# Evaluation plan

How to measure whether the recommender's rankings are *good* — without asking
humans to rank cities themselves. Written as a self-contained brief so a new
chat can implement this without backreading.

---

## What's already in place

- **Contrastive ranker:** held-out 1k synthetic split, **recall@10 = 0.804**
  (descriptive 0.890, numerical 0.837, anchored 0.758, multi-anchor 0.650).
  The "right answer" set is LLM-generated alongside training data. Lives in
  `backend/scripts/qualitative_eval.py`.
- **Classical ranker:** *no quantitative eval at all.* We have anecdotal
  examples in `README.md` ("Seattle but sunnier" → Amman 44.2%) but no metric
  that runs in CI or detects regressions.
- **No unified comparison.** Classical and contrastive can't be A/B'd against
  each other today. The compare-mode UI flag (`?compare=1`) exists for
  qualitative inspection only.

The Hogwarts → Cairo bug we just fixed is exhibit A for why this matters — a
silent semantic regression sat in the codebase undetected because nothing was
checking the *output* of the recommender against any expectation.

---

## Three approaches

The right answer is some mix of these. They're listed cheapest → most
expensive and catch progressively fuzzier failure modes.

### 1. Structural assertions (behavioral invariants)

A pytest suite of ~15–30 hand-picked queries, each with an assertion the
output must satisfy. Deterministic, no LLM, runs in milliseconds.

**Examples:**

| Query | Assertion |
|---|---|
| `"Tokyo but warmer winters"` | every top-5 city has winter-temp ≥ Tokyo's, within 1°C tolerance |
| `"Tokyo but colder winters"` | every top-5 city has winter-temp ≤ Tokyo's |
| `"like Tokyo"` (no modifier) | top-1 ∈ {Shanghai, Seoul, Osaka} (pre-computed sibling set) |
| `"Hogwarts but sunny"` | response has `anchor_error` set; `results == []` |
| `"Tokio but sunnier"` | resolved anchor name == "Tokyo" (typo correction sanity) |
| `"X but warmer"` vs `"X but colder"` (same X) | top-10 sets are ~disjoint (overlap ≤ 2) |

**Pros:** free, fast, fails loudly with a line number. Would have caught the
Hogwarts bug from day one.

**Cons:** doesn't tell you whether ranking *quality* is good — only whether
it's not catastrophically broken. Won't catch e.g. "Vienna at #4 should be
at #1." Brittle to canonical-set changes (new cities can break sibling sets).

---

### 2. Climatology-grounded synthetic metric (the workhorse)

Generalize approach #1 from a hand-picked list to a generated dataset, with
ground truth derived from the same σ-space the recommender operates in.

**Construction:**

1. LLM (gpt-4o-mini, constrained-decoding) generates ~500 query templates of
   the form `(anchor_city, axis, scope, intensity)` covering every reasonable
   combination across the 230 canonical cities.
2. For each query, compute the **expected delta vector** by running
   `vibe_table.apply_vibes()` on a zero vector — this gives the σ-space shift
   the ideal answer should embody on the *touched dims*.
3. Run the recommender, take top-K results.
4. Per-query score: cosine similarity between the **observed delta**
   (mean(top-K profiles in σ-space) − anchor profile in σ-space, restricted
   to touched dims) and the **expected delta** from step 2.
5. Aggregate: mean cosine across all queries, broken out by axis / scope /
   anchor-region.

**Critical property:** ground truth comes from the vibe table itself, not
from a model. If the recommender disagrees with the vibe table, the metric
reflects that. There's no human or LLM in the loop.

**What it catches:**

- Ranker drift (e.g. a similarity-metric change subtly inverts an axis)
- Per-axis weakness (e.g. "windier" queries score 0.3 while "warmer" score
  0.8 — tells you the wind dims need re-weighting)
- Classical-vs-contrastive comparison on the anchored subset (contrastive's
  weakness on `"X but Y"` queries should show up as a low score here, in a
  way the recall@10 metric doesn't capture)

**What it misses:**

- Free-form vibe queries with no anchor (no expected-delta to compare to)
- "The top-10 has the right *direction* but the wrong cities" — measures
  centroid alignment, not per-city correctness
- Anything that requires world knowledge ("Lima is a coastal city" — the
  metric doesn't know that)

**Effort:** ~half a day. We already have `vibe_table.py`, the canonical city
list, and the σ-space scaler. The triple-generation script for the
contrastive ranker is a near-fit and can be retargeted.

---

### 3. LLM-as-judge (top OpenAI model, scored against real data)

For the messy queries that #2 can't score (free-form vibe text, multi-anchor
blends, queries where world knowledge matters), have a strong model judge
the rankings.

**Non-obvious requirements** — the difference between "judge that works" and
"judge that reads astrology":

- **Feed the judge the actual climate numbers, not just city names.** Each
  candidate city's prompt entry must include annual temp, humidity,
  precipitation, sunshine, wind, etc. from the 96-d profile. Without this,
  the model riffs on prior knowledge ("Lima feels foggy") and you end up
  outsourcing ground truth to its training data. With it, scoring becomes a
  reasoning task over data you provided, which is what these models are
  actually good at.
- **Pairwise, not absolute.** Show query + ranker-A top-10 + ranker-B top-10
  (blinded, order randomized) and ask which is better, with a 1-sentence
  reason. Absolute 1–10 scores compress into a narrow band and drift
  run-to-run; pairwise win rates are stable and statistically tractable.
- **Pin a fixed baseline.** Always compare new rankers against a frozen
  reference (today's classical at commit `<sha>`). That way "judge says win
  rate went up" is a real signal, not just judge mood.
- **Confidence intervals.** Bootstrap N=100 over the query set; report
  win-rate ± 95% CI. A 52% vs 48% split with wide CI is noise, not progress.

**Cost note:** ~$0.10/query at gpt-5-class pairwise scoring × 200 queries ×
2 rankers = ~$40 per full eval run. Acceptable for milestones, not for every
push.

**What it catches:** the qualitative tail. "These results look reasonable
but feel weird." "The top result is technically right but a less-famous
better match exists." Things you can't write a closed-form metric for.

**What it misses:** mechanical regressions cheaply. The Hogwarts bug *would*
be caught by a judge, but at $0.10 per call vs 5ms for a pytest assertion.
Don't lean on the judge for things assertions can do.

---

## If we go LLM-judge-only

The user has top-tier OpenAI access and asked whether a strong model judge
alone is sufficient. Short answer: **yes, with three constraints** —

1. Feed the judge real climate stats, not just city names (see above).
2. Pairwise scoring against a pinned baseline, not absolute.
3. Keep a tiny ~10-assertion pytest suite alongside it for the failure paths
   (Hogwarts case, monotonicity, anchor-resolution sanity). It's free, it
   runs on every push, and it fails with a stack trace. The judge handles
   ranking quality; assertions handle correctness.

What you give up by skipping #2 (the synthetic metric):

- Per-axis breakdown ("the recommender is weak on `windier` queries
  specifically"). The judge can tell you the *whole* ranking is worse, but
  not why.
- A cheap signal to run on every PR. The judge is too slow/expensive for
  per-commit feedback; it's a milestone tool.

If the choice is "one of these, not all three," the layered minimum is
**(small pytest suite) + (LLM judge with real climate data, pairwise vs
pinned baseline)**. The synthetic metric is the highest-leverage *addition*
but isn't strictly required if budget is fine.

---

## Recommendation

Build in this order:

1. **Pytest assertion suite** — half a day. ~15 queries covering
   monotonicity, anchor failure paths, typo correction, and known sibling
   sets. Lives in `backend/tests/test_recommender_invariants.py`. Runs in CI
   on every push.
2. **LLM-as-judge harness** — one to two days. Pairwise comparison script
   that takes two ranker configs (e.g. classical vs contrastive, or current
   vs proposed change) and outputs win-rate with CI. Always includes the
   per-city climate stats in the judge prompt. Lives in
   `backend/scripts/judge_eval.py`.
3. **Climatology-grounded synthetic metric** — half a day, after #1 and #2
   are running. Gives you the per-axis breakdown the judge can't, at zero
   marginal cost per run.

Total: 2–3 days of work for a real evaluation pipeline that catches
mechanical regressions, ranks ranker changes quantitatively, and gives
per-axis insight.

---

## Open questions for the next session

- **Judge model choice** — gpt-5? o3? gpt-4o? The reasoning models are
  better at this kind of grounded comparison but slower. Worth a small
  consistency study (run same N=20 pairwise on each, check agreement rate).
- **Baseline pinning** — what counts as "today's classical"? Probably commit
  `6351a22` (post-Hogwarts-fix `main`). Need to snapshot its outputs.
- **Free-form vibe query generation** — for the judge eval, where do queries
  come from? Probably a mix of (a) the example queries in `README.md`, (b)
  LLM-generated free-form vibes, (c) past tournament picks once those start
  being logged.
- **Should we use the existing 10k synthetic triples?** They were built for
  contrastive *training*, where positives/negatives are city pairs. Repurposing
  them for ranker eval needs care — recall@10 against synthetic positives is
  what the contrastive ranker was *fit to*, so the 0.804 number partly reflects
  fit, not generalization. The judge eval should use a fresh query set.
- **CI integration** — pytest suite goes in CI immediately; judge eval is too
  expensive for per-PR. Probably a manual `make eval` target plus a weekly
  scheduled run on `main`.

---

## Pointers (for the new chat)

- Vibe → σ-space delta mapping: `backend/services/vibe_table.py`
- Anchor resolution + the new `anchor_error` path: `backend/services/recommender.py`
- StandardScaler + canonical profiles: `backend/services/state.py`
- Existing contrastive eval (recall@k, t-SNE): `backend/scripts/qualitative_eval.py`
- Synthetic triple generation (template for #2 and #3): `backend/scripts/generate_triples.py`
- Canonical 230-city list: `backend/vendor/weather_ml/cities.py`
- Frontend type for ranker output: `frontend/lib/api.ts` (`RecommendResponse`)
