"""Precompute frozen MiniLM forward over the labeled queries once, save to
text_embeddings.npz. The training loop reads embeddings instead of running
the 22M-param encoder every step.

    python -m backend.scripts.precompute_embeddings \\
        --in backend/cache/contrastive_triples.jsonl \\
        --out backend/cache/text_embeddings.npz

Output schema:
    queries:    (N,) object array of the exact query strings
    embeddings: (N, 384) float32 — sentence-transformers/all-MiniLM-L6-v2

The training script joins triples to embeddings by exact query string.
Resumable in the sense of "skip if output exists and matches input"; no
partial-resume — re-runs the full forward pass since it's seconds on a GPU
and 1-3 min on CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


def load_unique_queries(jsonl_path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row["query"]
            if q not in seen:
                seen.add(q)
                out.append(q)
    return out


def encode_all(queries: list[str], batch_size: int) -> np.ndarray:
    # Lazy import: lets `--help` work without the dep installed.
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"encoding {len(queries)} queries (batch_size={batch_size})", flush=True)
    embs = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # the contrastive head normalizes downstream
    )
    return embs.astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="JSONL with a 'query' field per row")
    p.add_argument("--out", type=Path, required=True,
                   help="output .npz path")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = p.parse_args()

    queries = load_unique_queries(args.in_path)
    print(f"loaded {len(queries)} unique queries from {args.in_path}")

    embeddings = encode_all(queries, args.batch_size)
    print(f"embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        queries=np.array(queries, dtype=object),
        embeddings=embeddings,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
