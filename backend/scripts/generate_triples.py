"""Generate synthetic (query, positives, negatives) triples for contrastive training.

Two-stage OpenAI pipeline. Both stages are resumable — re-run the same command
and rows already in the output file are skipped.

Workflow:

    # 1. Trial run: generate 200 queries, eyeball, tighten the prompt if needed.
    python backend/scripts/generate_triples.py queries \\
        --n 200 --out backend/cache/trial_queries.jsonl
    python backend/scripts/generate_triples.py review \\
        --in backend/cache/trial_queries.jsonl --n 50

    # 2. Full query set.
    python backend/scripts/generate_triples.py queries \\
        --n 10000 --out backend/cache/queries.jsonl

    # 3. Label each query with 3 positives + 5 negatives from the canonical 230.
    python backend/scripts/generate_triples.py label \\
        --in backend/cache/queries.jsonl \\
        --out backend/cache/contrastive_triples.jsonl

    # 4. Spot-check.
    python backend/scripts/generate_triples.py review \\
        --in backend/cache/contrastive_triples.jsonl --n 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.vendor.weather_ml.cities import CITIES  # noqa: E402

CITY_NAMES: tuple[str, ...] = tuple(c["name"] for c in CITIES)
CityName = Literal[CITY_NAMES]  # OpenAI structured output rejects off-list names

QUERY_BATCH_SIZE = 25


class QueryBatch(BaseModel):
    queries: list[str] = Field(min_length=1)


class TripleLabels(BaseModel):
    positives: list[CityName] = Field(min_length=3, max_length=3)
    negatives: list[CityName] = Field(min_length=5, max_length=5)


class BatchItem(BaseModel):
    query: str
    positives: list[CityName] = Field(min_length=3, max_length=3)
    negatives: list[CityName] = Field(min_length=5, max_length=5)


class BatchLabels(BaseModel):
    items: list[BatchItem] = Field(min_length=1)


LABEL_BATCH_SIZE = 5


QUERY_SYSTEM_PROMPT = """You generate realistic free-text search queries for a natural-language climate recommender. Real users type these queries when they want to find a city to live in, move to, or visit, based on its weather.

Generate exactly {n} diverse, substantive queries. Vary phrasing across these styles, mixing them roughly evenly:
- Comparative: "Seattle but sunnier", "like Tokyo but drier"
- Descriptive: "warm sunny coastal city", "mild damp climate year round"
- Preference: "I want hot dry summers and cold snowy winters"
- Multi-anchor: "between Barcelona and Miami", "like Tokyo but more European"
- Loose comparison: "European cities that feel like San Francisco"

Hard rules:
- 3-15 words. Plain English. Lowercase preferred. No quotes, no numbering, no commentary.
- Each query MUST contain a clear signal in one or more of: temperature, humidity, rainfall, sunshine, wind, seasonality, dryness/wetness.
- EVERY modifier word in the query must describe a weather attribute. Allowed modifier words include: warmer, colder, milder, harsher, wetter, drier, sunnier, cloudier, windier, calmer, more/less humid, more/less seasonal, hotter summers, colder winters, etc.
- Cover diverse climate niches across batches: tropical, desert, mediterranean, oceanic, continental, polar, monsoon, alpine, equatorial.
- Use real city/region names if you mention any.
- Avoid duplicates and near-duplicates within this batch.

Forbidden — do NOT mention or use as a modifier:
- Cost of living, prices, taxes, affordability, expensive, cheap
- Walkability, transit, infrastructure, accessible, accessibility, remote
- Food, language, culture, politics, religion, demographics
- Population, density, spacious, crowded, big, small (as a city descriptor), popular, touristy, exclusive
- Specific calendar months/dates or events
- Scenery descriptors: pretty, scenic, beautiful, picturesque
- Anything not derivable from weather data alone.

If you find yourself reaching for a non-climate modifier, drop it or replace it with a climate one."""


SEED_TOPICS = [
    "tropical climates and humid heat",
    "deserts, dryness, and clear skies",
    "mediterranean climates with mild winters",
    "oceanic and maritime climates",
    "continental climates with strong four-season swing",
    "alpine, mountain, and high-elevation places",
    "monsoon and wet-season rainfall patterns",
    "polar, subarctic, and brutally cold places",
    "equatorial year-round warmth and steady weather",
    "windy, stormy coastal places",
    "comparative queries naming a US city as the anchor",
    "comparative queries naming a European city as the anchor",
    "comparative queries naming an Asian city as the anchor",
    "comparative queries naming a Latin American city as the anchor",
    "preference statements about ideal personal weather",
    "loose cross-continent comparisons",
    "multi-city blends and contrasts",
    "queries focused on a single season's character",
    "everyday vague climate wishes from non-experts",
    "queries from someone allergic to humidity / heat / cold",
    "queries with concrete numerical thresholds (e.g. winter highs above 50F, under 600mm of rain per year, summer humidity below 50 percent, snow more than 100 days per year)",
]


LABEL_SYSTEM_PROMPT = """You score cities against a batch of climate queries in a single response.

For EACH input query, return one item containing:
- query: the exact query string from the input (for matching)
- positives: the 3 cities that BEST match this query, ranked best to worst
- negatives: 5 cities whose climate diverges sharply from what this query asks for — clearly bad suggestions that would feel jarring

Choose cities only from the provided candidate list. If a city named in a query is in the candidate list (e.g. "Seattle" in "Seattle but sunnier"), include it in positives only if it really is among the top matches considering the modification — otherwise skip it.

You MUST return exactly one item per input query. Echo the query string verbatim in each item so the caller can match results to inputs. Do not skip queries, do not invent queries, do not merge queries.

Reason about: temperature, humidity, rainfall, sunshine, wind, seasonality. Use real climatology. Pick negatives that are clearly wrong on the dimensions the query cares about, not merely mediocre matches."""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_query(q: str) -> str:
    return " ".join(q.strip().lower().split())


async def gen_query_batch(client: AsyncOpenAI, n: int, seed_topic: str) -> list[str]:
    completion = await client.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT.format(n=n)},
            {
                "role": "user",
                "content": (
                    f"Generate {n} queries. Bias this batch toward the following theme to keep "
                    f"variety high across batches, but don't restrict yourself to it entirely: {seed_topic}"
                ),
            },
        ],
        response_format=QueryBatch,
        temperature=1.0,
    )
    parsed = completion.choices[0].message.parsed
    return list(parsed.queries) if parsed else []


async def run_query_generation(out_path: Path, n_target: int, concurrency: int) -> None:
    existing = load_jsonl(out_path)
    seen = {normalize_query(r["query"]) for r in existing}
    print(f"resuming with {len(seen)} existing queries; target {n_target}")
    if len(seen) >= n_target:
        print("target already met")
        return

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    sem = asyncio.Semaphore(concurrency)

    topic_rotation = list(SEED_TOPICS)
    random.Random(0).shuffle(topic_rotation)

    async def one_call(i: int) -> list[str] | Exception:
        async with sem:
            topic = topic_rotation[i % len(topic_rotation)]
            try:
                return await gen_query_batch(client, QUERY_BATCH_SIZE, topic)
            except Exception as e:
                return e

    call_id = 0
    consecutive_dead_waves = 0
    while len(seen) < n_target:
        wave_size = max(concurrency, 5)
        wave_ids = list(range(call_id, call_id + wave_size))
        call_id += wave_size
        results = await asyncio.gather(*[one_call(i) for i in wave_ids])
        batches = [r for r in results if isinstance(r, list)]
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"  {len(errors)}/{len(results)} calls failed; first error: {errors[0]}")
        if not batches:
            consecutive_dead_waves += 1
            if consecutive_dead_waves >= 2:
                raise RuntimeError(
                    f"two consecutive waves of {wave_size} calls all failed — "
                    f"aborting. Fix the underlying issue (likely API key or rate limit) "
                    f"and re-run; progress so far is saved to {out_path}"
                )
            continue
        consecutive_dead_waves = 0
        new_rows: list[dict] = []
        for batch in batches:
            for q in batch:
                key = normalize_query(q)
                if key and key not in seen:
                    seen.add(key)
                    new_rows.append({"query": q.strip()})
                    if len(seen) >= n_target:
                        break
            if len(seen) >= n_target:
                break
        append_jsonl(out_path, new_rows)
        print(f"  +{len(new_rows)} new (total {len(seen)} / {n_target})")


CANDIDATE_LIST_TEXT = ", ".join(CITY_NAMES)


async def label_batch(client: AsyncOpenAI, batch: list[str]) -> list[dict]:
    """Label one batch of queries in a single API call. Returns one dict per
    successfully matched query; partial failures (mismatched/missing) are
    silently dropped, and the caller's resume logic re-attempts them."""
    queries_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(batch))
    user_msg = (
        f"Score these {len(batch)} queries against the candidate list. "
        f"Return one item per query, echoing the query string verbatim.\n\n"
        f"Queries:\n{queries_text}\n\n"
        f"Candidate cities ({len(CITY_NAMES)}):\n{CANDIDATE_LIST_TEXT}"
    )
    try:
        completion = await client.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format=BatchLabels,
            temperature=0.3,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            return []
        by_query = {normalize_query(item.query): item for item in parsed.items}
        out: list[dict] = []
        for q in batch:
            item = by_query.get(normalize_query(q))
            if item is None:
                continue
            out.append({
                "query": q,
                "positives": list(item.positives),
                "negatives": list(item.negatives),
            })
        return out
    except Exception as e:
        print(f"  batch failed ({len(batch)} queries): {e}")
        return []


async def run_labeling(in_path: Path, out_path: Path, concurrency: int) -> None:
    queries_in = [r["query"] for r in load_jsonl(in_path)]
    done_set = {normalize_query(r["query"]) for r in load_jsonl(out_path)}
    todo = [q for q in queries_in if normalize_query(q) not in done_set]
    print(f"to label: {len(todo)} (skipping {len(queries_in) - len(todo)} already done)")
    if not todo:
        return

    batches = [todo[i: i + LABEL_BATCH_SIZE] for i in range(0, len(todo), LABEL_BATCH_SIZE)]
    print(f"  -> {len(batches)} batches of up to {LABEL_BATCH_SIZE} queries")

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    sem = asyncio.Semaphore(concurrency)

    async def one(batch: list[str]):
        async with sem:
            return await label_batch(client, batch)

    WAVE = max(concurrency * 5, 25)
    consecutive_dead_waves = 0
    attempted = 0
    succeeded = 0
    for wave_start in range(0, len(batches), WAVE):
        wave = batches[wave_start: wave_start + WAVE]
        results = await asyncio.gather(*[one(b) for b in wave])
        rows = [r for batch_rows in results for r in batch_rows]
        append_jsonl(out_path, rows)
        wave_attempted = sum(len(b) for b in wave)
        attempted += wave_attempted
        succeeded += len(rows)
        print(f"  labeled {attempted} / {len(todo)}  (+{len(rows)}/{wave_attempted} this wave, total succeeded {succeeded})")
        if not rows:
            consecutive_dead_waves += 1
            if consecutive_dead_waves >= 2:
                raise RuntimeError(
                    f"two consecutive waves of {len(wave)} batches produced "
                    f"no rows — aborting. Fix the underlying issue and re-run; "
                    f"progress so far is saved to {out_path}"
                )
        else:
            consecutive_dead_waves = 0


def run_review(in_path: Path, n: int, seed: int | None) -> None:
    rows = load_jsonl(in_path)
    print(f"file has {len(rows)} rows")
    if not rows:
        return
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    has_labels = "positives" in rows[0]
    for i, r in enumerate(sample, 1):
        if has_labels:
            print(f"\n[{i:>3}] {r['query']}")
            print(f"      + {', '.join(r['positives'])}")
            print(f"      - {', '.join(r['negatives'])}")
        else:
            print(f"[{i:>3}] {r['query']}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("queries", help="generate diverse query strings")
    q.add_argument("--n", type=int, required=True, help="target number of unique queries")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--concurrency", type=int, default=20)

    lbl = sub.add_parser("label", help="label queries with positives + negatives")
    lbl.add_argument("--in", dest="in_path", type=Path, required=True)
    lbl.add_argument("--out", type=Path, required=True)
    lbl.add_argument("--concurrency", type=int, default=20)

    rv = sub.add_parser("review", help="print N random samples for spot-check")
    rv.add_argument("--in", dest="in_path", type=Path, required=True)
    rv.add_argument("--n", type=int, default=50)
    rv.add_argument("--seed", type=int, default=None)

    args = p.parse_args()
    if args.cmd == "queries":
        asyncio.run(run_query_generation(args.out, args.n, args.concurrency))
    elif args.cmd == "label":
        asyncio.run(run_labeling(args.in_path, args.out, args.concurrency))
    elif args.cmd == "review":
        run_review(args.in_path, args.n, args.seed)


if __name__ == "__main__":
    main()
