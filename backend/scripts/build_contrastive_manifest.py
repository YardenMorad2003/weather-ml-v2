"""Build the static manifest served by GET /contrastive/manifest.

The deployed backend does not depend on torch; the .pt state dicts are read
once here (locally, in the training venv) and re-emitted as JSON so the API
can serve them with numpy + json only.

    python -m backend.scripts.build_contrastive_manifest \\
        --cache-dir backend/cache \\
        --out backend/cache/contrastive_manifest.json

Output schema (matches CONTRASTIVE_DEPLOY.md):
    {
      "cities":     [str, ... len 230],
      "embeddings": [[float32 x 32] x 230],   # L2-normalized
      "projection": {"weight": [[...] x 32 of len 384], "bias": [float x 32]},
      "scaler":     {"mean": [float x 96], "scale": [float x 96]},
      "embed_dim":  32,
      "text_dim":   384,
      "city_input_dim": 96
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build_manifest(cache_dir: Path) -> dict:
    emb_npz = np.load(cache_dir / "city_embeddings_learned.npz", allow_pickle=True)
    cities = [str(c) for c in emb_npz["cities"].tolist()]
    embeddings = emb_npz["embeddings"].astype(np.float32)
    if embeddings.shape != (len(cities), 32):
        raise ValueError(
            f"unexpected embeddings shape {embeddings.shape}, expected ({len(cities)}, 32)"
        )

    scaler_npz = np.load(cache_dir / "city_scaler.npz", allow_pickle=True)
    scaler_mean = scaler_npz["mean"].astype(np.float32)
    scaler_scale = scaler_npz["scale"].astype(np.float32)

    proj_state = torch.load(
        cache_dir / "text_proj_state.pt", map_location="cpu", weights_only=True
    )
    proj_weight = proj_state["proj.weight"].cpu().numpy().astype(np.float32)
    proj_bias = proj_state["proj.bias"].cpu().numpy().astype(np.float32)
    if proj_weight.shape != (32, 384):
        raise ValueError(f"unexpected proj.weight shape {proj_weight.shape}")
    if proj_bias.shape != (32,):
        raise ValueError(f"unexpected proj.bias shape {proj_bias.shape}")

    return {
        "cities": cities,
        "embeddings": embeddings.tolist(),
        "projection": {
            "weight": proj_weight.tolist(),
            "bias": proj_bias.tolist(),
        },
        "scaler": {
            "mean": scaler_mean.tolist(),
            "scale": scaler_scale.tolist(),
        },
        "embed_dim": 32,
        "text_dim": 384,
        "city_input_dim": 96,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("backend/cache"))
    ap.add_argument(
        "--out", type=Path, default=Path("backend/cache/contrastive_manifest.json")
    )
    args = ap.parse_args()

    manifest = build_manifest(args.cache_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))
    size_kb = args.out.stat().st_size / 1024
    print(
        f"wrote {args.out} ({size_kb:.1f} KB) "
        f"cities={len(manifest['cities'])} embeddings={len(manifest['embeddings'])}x{len(manifest['embeddings'][0])}"
    )


if __name__ == "__main__":
    main()
