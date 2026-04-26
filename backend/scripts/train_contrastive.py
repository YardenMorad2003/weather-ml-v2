"""Train the dual-encoder contrastive model on the synthetic LLM-labeled triples.

Pipeline:
    1. Load 230 raw city profiles, phase-align, StandardScale.
    2. Load 10k labeled (query, positives, negatives) triples + precomputed
       MiniLM (384-d) text embeddings for each query.
    3. 90/10 split by query, fixed seed.
    4. Symmetric InfoNCE with in-batch negatives (CLIP-style); batch=256.
    5. Save trained CityEncoder + TextProjection + StandardScaler params.
    6. Save the final (230, 32) L2-normalized city embedding cache.
    7. Print recall@10 on held-out queries (gold = the 3 LLM-labeled positives).

Usage (after `precompute_embeddings.py`):

    python -m backend.scripts.train_contrastive \\
        --triples backend/cache/contrastive_triples.jsonl \\
        --text-emb backend/cache/text_embeddings.npz \\
        --profiles backend/cache/profiles_2023-01-01_2024-12-31.npz \\
        --out-dir backend/cache \\
        --epochs 100 --batch-size 256 --lr 1e-3 --seed 42

Auto-detects CUDA. On Colab T4: ~30s. On CPU: ~2-5 min.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Imports below this line need REPO_ROOT on path
from backend.services.contrastive import (  # noqa: E402
    CityEncoder, TextProjection, info_nce_loss,
    CITY_INPUT_DIM, EMBED_DIM, TEXT_INPUT_DIM,
)
from backend.services.profile import phase_align  # noqa: E402
from backend.vendor.weather_ml.cities import CITIES  # noqa: E402


CITY_NAMES: list[str] = [c["name"] for c in CITIES]
CITY_NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(CITY_NAMES)}


def load_triples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_text_embeddings(path: Path) -> tuple[dict[str, int], np.ndarray]:
    """Returns (query -> row index, (N, 384) embeddings)."""
    data = np.load(path, allow_pickle=True)
    queries = list(data["queries"])
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    q2i = {q: i for i, q in enumerate(queries)}
    return q2i, embeddings


def load_city_profiles_scaled(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw (230, 96) profiles, phase-align, StandardScale.

    Returns (scaled_profiles, scaler_mean, scaler_scale). The scaler params
    must travel with the encoder so inference can reproduce the same input
    space. We avoid sklearn at inference; this is just (x - mean) / scale.
    """
    raw = np.load(path)["profiles"]
    aligned = np.stack([phase_align(raw[i]) for i in range(raw.shape[0])])
    mean = aligned.mean(axis=0)
    std = aligned.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # constant features -> identity
    scaled = (aligned - mean) / std
    return scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_train_pairs(
    triples: list[dict],
    q2i: dict[str, int],
) -> list[tuple[int, list[int]]]:
    """Returns [(query_emb_idx, [pos_city_idx, ...]), ...]. Skips rows whose
    query/cities aren't found (defensive — both sets are validated upstream
    via Pydantic Literal so this should be empty in practice)."""
    out: list[tuple[int, list[int]]] = []
    skipped = 0
    for t in triples:
        q_idx = q2i.get(t["query"])
        if q_idx is None:
            skipped += 1
            continue
        pos_idxs = [CITY_NAME_TO_IDX[n] for n in t["positives"] if n in CITY_NAME_TO_IDX]
        if not pos_idxs:
            skipped += 1
            continue
        out.append((q_idx, pos_idxs))
    if skipped:
        print(f"  warning: skipped {skipped} triples (missing query or city)")
    return out


def split_by_query(
    pairs: list[tuple[int, list[int]]],
    test_frac: float,
    seed: int,
) -> tuple[list, list]:
    rng = np.random.RandomState(seed)
    n = len(pairs)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_frac))
    test_idx = set(idx[:n_test].tolist())
    train, test = [], []
    for i, p in enumerate(pairs):
        (test if i in test_idx else train).append(p)
    return train, test


def train_one_epoch(
    text_enc: TextProjection,
    city_enc: CityEncoder,
    optimizer,
    text_emb: "torch.Tensor",
    city_profiles: "torch.Tensor",
    pairs: list[tuple[int, list[int]]],
    batch_size: int,
    temperature: float,
    rng: np.random.RandomState,
) -> float:
    import torch
    text_enc.train()
    city_enc.train()
    rng.shuffle(pairs)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start: start + batch_size]
        if len(batch) < 2:
            continue  # InfoNCE needs at least 2 for in-batch negatives
        q_idx = [p[0] for p in batch]
        # Sample one positive per query from its 3 listed positives
        pos_idx = [p[1][rng.randint(0, len(p[1]))] for p in batch]

        q_in = text_emb[q_idx]
        c_in = city_profiles[pos_idx]
        q_out = text_enc(q_in)
        c_out = city_enc(c_in)
        loss = info_nce_loss(q_out, c_out, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def recall_at_k(
    text_enc: TextProjection,
    city_enc: CityEncoder,
    text_emb: "torch.Tensor",
    city_profiles: "torch.Tensor",
    pairs: list[tuple[int, list[int]]],
    k: int = 10,
) -> float:
    """Average recall@k over the held-out queries.

    For each query: project the text embedding, cosine-rank all 230 city
    embeddings, mark a hit if any gold positive is in top-k. Returns the
    fraction of queries with at least one hit. Caller is expected to wrap
    in torch.no_grad() (this fn is called inside such a block at every
    eval site, so we don't decorate it here).
    """
    import torch
    text_enc.eval()
    city_enc.eval()

    # Encode all cities once
    c_emb = city_enc(city_profiles)
    c_emb = torch.nn.functional.normalize(c_emb, dim=1)  # (n_cities, D)

    hits = 0
    for q_idx, pos_idxs in pairs:
        q_out = text_enc(text_emb[q_idx: q_idx + 1])
        q_out = torch.nn.functional.normalize(q_out, dim=1)  # (1, D)
        sims = (q_out @ c_emb.T).squeeze(0)                  # (n_cities,)
        top = torch.topk(sims, k=k).indices.tolist()
        if any(p in top for p in pos_idxs):
            hits += 1
    return hits / max(len(pairs), 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--triples", type=Path, required=True)
    p.add_argument("--text-emb", type=Path, required=True)
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10,
                   help="run held-out recall@10 every N epochs; 0 to skip")
    args = p.parse_args()

    # Heavy imports deferred until after argparse so --help is fast.
    import torch
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading data...")
    triples = load_triples(args.triples)
    print(f"  triples: {len(triples)}")
    q2i, text_emb_np = load_text_embeddings(args.text_emb)
    print(f"  text embeddings: {text_emb_np.shape}")
    assert text_emb_np.shape[1] == TEXT_INPUT_DIM
    city_scaled_np, scaler_mean, scaler_scale = load_city_profiles_scaled(args.profiles)
    print(f"  city profiles: {city_scaled_np.shape}")
    assert city_scaled_np.shape == (len(CITIES), CITY_INPUT_DIM), \
        f"profiles shape {city_scaled_np.shape} != ({len(CITIES)}, {CITY_INPUT_DIM})"

    pairs = build_train_pairs(triples, q2i)
    print(f"  usable (query, positives) pairs: {len(pairs)}")

    train_pairs, test_pairs = split_by_query(pairs, args.test_frac, args.seed)
    print(f"  train / test split: {len(train_pairs)} / {len(test_pairs)}")

    text_emb = torch.from_numpy(text_emb_np).to(device)
    city_profiles = torch.from_numpy(city_scaled_np).to(device)

    text_enc = TextProjection().to(device)
    city_enc = CityEncoder().to(device)
    n_params = sum(p.numel() for p in text_enc.parameters() if p.requires_grad) + \
               sum(p.numel() for p in city_enc.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(text_enc.parameters()) + list(city_enc.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rng = np.random.RandomState(args.seed)

    print(f"\ntraining for {args.epochs} epochs (batch={args.batch_size}, "
          f"lr={args.lr}, T={args.temperature})")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            text_enc, city_enc, optimizer,
            text_emb, city_profiles, train_pairs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            rng=rng,
        )
        log = f"epoch {epoch:>3}/{args.epochs}  loss {avg_loss:.4f}"
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            with torch.no_grad():
                r10 = recall_at_k(text_enc, city_enc, text_emb, city_profiles, test_pairs, k=10)
                r1 = recall_at_k(text_enc, city_enc, text_emb, city_profiles, test_pairs, k=1)
                log += f"  | held-out recall@1 {r1:.3f}  recall@10 {r10:.3f}"
        print(log, flush=True)
    elapsed = time.time() - t0
    print(f"\ntrained in {elapsed:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(text_enc.state_dict(), args.out_dir / "text_proj_state.pt")
    torch.save(city_enc.state_dict(), args.out_dir / "city_encoder_state.pt")
    print(f"  wrote {args.out_dir / 'text_proj_state.pt'}")
    print(f"  wrote {args.out_dir / 'city_encoder_state.pt'}")

    np.savez(
        args.out_dir / "city_scaler.npz",
        mean=scaler_mean,
        scale=scaler_scale,
    )
    print(f"  wrote {args.out_dir / 'city_scaler.npz'}")

    # Final L2-normalized city embeddings for inference
    with torch.no_grad():
        city_enc.eval()
        c_emb = city_enc(city_profiles)
        c_emb = torch.nn.functional.normalize(c_emb, dim=1).cpu().numpy().astype(np.float32)
    np.savez(
        args.out_dir / "city_embeddings_learned.npz",
        cities=np.array(CITY_NAMES, dtype=object),
        embeddings=c_emb,
    )
    print(f"  wrote {args.out_dir / 'city_embeddings_learned.npz'}  shape={c_emb.shape}")

    # Persist the test-split queries so a separate eval script can reproduce
    test_queries = [
        {"query": next(q for q, i in q2i.items() if i == q_idx),
         "positives": [CITY_NAMES[i] for i in pos_idxs]}
        for q_idx, pos_idxs in test_pairs
    ]
    with (args.out_dir / "contrastive_test_split.jsonl").open("w", encoding="utf-8") as f:
        for r in test_queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {args.out_dir / 'contrastive_test_split.jsonl'}")

    print("\nfinal held-out metrics:")
    with torch.no_grad():
        r1 = recall_at_k(text_enc, city_enc, text_emb, city_profiles, test_pairs, k=1)
        r5 = recall_at_k(text_enc, city_enc, text_emb, city_profiles, test_pairs, k=5)
        r10 = recall_at_k(text_enc, city_enc, text_emb, city_profiles, test_pairs, k=10)
    print(f"  recall@1  {r1:.3f}")
    print(f"  recall@5  {r5:.3f}")
    print(f"  recall@10 {r10:.3f}  (target >0.70)")


if __name__ == "__main__":
    main()
