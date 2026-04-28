"""Sigma-space eval extension for no-anchor queries — both rankers.

Two adaptations of the EVAL_PLAN approach #2 metric:

  1. CLASSICAL no-anchor path (recommender.py:94-95): anchor = dataset centroid,
     expected_delta = apply_vibes(zero_vec, vibes). Classical recommender ranks
     by weighted-Euclidean toward (centroid + delta).

  2. CONTRASTIVE ranker: encode the natural-language query via MiniLM ->
     text_proj -> cosine vs the trained 230x32 city embeddings. Doesn't use
     the vibe table at all - it works on the query text directly. We score it
     against the SAME structured-vibes expected_delta to see whether the
     text-trained ranker lands its top-K in the right sigma-space region.

Each query is given as (text, structured_vibes) so we can run the contrastive
model on the text and the classical model on the structured form, then score
both with the same closed-form ground truth - no LLM parser involved at eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.state import get_state  # noqa: E402
from backend.services.vibe_table import apply_vibes  # noqa: E402
from backend.services.contrastive import CityEncoder, TextProjection  # noqa: E402
from backend.vendor.weather_ml.cities import CITIES  # noqa: E402


FOCUS_WEIGHT = 6.0
BASE_WEIGHT = 1.0
EPS = 1e-9
FEATURE_NAMES = [
    "temp", "humidity", "dewpoint", "precip",
    "cloud", "pressure", "wind", "clear_sky",
]
CITY_NAMES = [c["name"] for c in CITIES]
CACHE = REPO_ROOT / "backend" / "cache"


def summarize_delta(delta_96, touched_96):
    delta = delta_96.reshape(12, 8)
    touched = touched_96.reshape(12, 8)
    parts = []
    for f in range(8):
        mask = touched[:, f] > 0
        if not mask.any():
            continue
        avg = float(delta[mask, f].mean())
        sign = "+" if avg >= 0 else "-"
        parts.append(f"{FEATURE_NAMES[f]} {sign}{abs(avg):.2f}sigma")
    return ", ".join(parts) if parts else "(none)"


def sigma_score(observed_delta, expected_delta):
    exp_norm = float(np.linalg.norm(expected_delta))
    obs_norm = float(np.linalg.norm(observed_delta))
    cos = float(np.dot(expected_delta, observed_delta)
                / (exp_norm * obs_norm + EPS))
    return cos, exp_norm, obs_norm


def classical_no_anchor_topk(profiles_scaled, vibes, top_k=10):
    """Mirror recommender.py path with anchor=None: start from centroid."""
    centroid_scaled = profiles_scaled.mean(axis=0)
    user_scaled = centroid_scaled.copy()
    user_modified, touched = apply_vibes(user_scaled, vibes)

    weights = BASE_WEIGHT + (FOCUS_WEIGHT - BASE_WEIGHT) * touched
    diffs = profiles_scaled - user_modified[None, :]
    dists = np.sqrt((weights[None, :] * diffs * diffs).sum(axis=1))
    order = np.argsort(dists).tolist()[:top_k]

    expected_delta = (user_modified - user_scaled) * touched
    observed_centroid = profiles_scaled[order].mean(axis=0)
    observed_delta = (observed_centroid - centroid_scaled) * touched
    return order, expected_delta, observed_delta, touched


def contrastive_topk(query_text, st_model, text_proj, city_embs, top_k=10):
    raw = st_model.encode([query_text], convert_to_numpy=True,
                          normalize_embeddings=False).astype(np.float32)
    with torch.no_grad():
        proj = F.normalize(text_proj(torch.from_numpy(raw)), dim=1).numpy()
    sims = (proj @ city_embs.T)[0]
    order = np.argsort(-sims).tolist()[:top_k]
    return order


def score_external_topk(top_idxs, profiles_scaled, vibes):
    """Apply the SAME sigma-space metric to an externally-supplied top-K
    (the contrastive ranker's picks). Anchor = centroid."""
    centroid_scaled = profiles_scaled.mean(axis=0)
    user_scaled = centroid_scaled.copy()
    user_modified, touched = apply_vibes(user_scaled, vibes)
    expected_delta = (user_modified - user_scaled) * touched
    observed_centroid = profiles_scaled[top_idxs].mean(axis=0)
    observed_delta = (observed_centroid - centroid_scaled) * touched
    return expected_delta, observed_delta, touched


def main():
    print("loading state (sigma-space profiles)...", flush=True)
    st = get_state()
    profiles_scaled = st["profiles_scaled"]
    print(f"  {len(profiles_scaled)} cities, dim {profiles_scaled.shape[1]}")

    print("loading contrastive artifacts...", flush=True)
    text_proj = TextProjection()
    text_proj.load_state_dict(torch.load(CACHE / "text_proj_state.pt", map_location="cpu"))
    text_proj.eval()
    emb_data = np.load(CACHE / "city_embeddings_learned.npz", allow_pickle=True)
    city_embs = emb_data["embeddings"]
    assert list(emb_data["cities"]) == CITY_NAMES
    print(f"  text_proj loaded; city_embs {city_embs.shape}")

    print("loading sentence-transformer (MiniLM)...", flush=True)
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    print("  done\n")

    cases = [
        ("tropical and humid",
         [{"axis": "warmer", "scope": "year_round", "intensity": "noticeably"},
          {"axis": "more_humid", "scope": "year_round", "intensity": "noticeably"}]),
        ("polar and dry",
         [{"axis": "colder", "scope": "year_round", "intensity": "noticeably"},
          {"axis": "drier", "scope": "year_round", "intensity": "noticeably"}]),
        ("hot dry desert climate",
         [{"axis": "warmer", "scope": "year_round", "intensity": "noticeably"},
          {"axis": "drier", "scope": "year_round", "intensity": "noticeably"}]),
        ("wet and cloudy place",
         [{"axis": "wetter", "scope": "year_round", "intensity": "noticeably"},
          {"axis": "cloudier", "scope": "year_round", "intensity": "noticeably"}]),
        ("cool windy maritime climate",
         [{"axis": "colder", "scope": "year_round", "intensity": "slightly"},
          {"axis": "windier", "scope": "year_round", "intensity": "noticeably"}]),
        ("sunny and dry summers",
         [{"axis": "sunnier", "scope": "summer", "intensity": "noticeably"},
          {"axis": "drier", "scope": "summer", "intensity": "noticeably"}]),
        ("mild winters with little rain",
         [{"axis": "warmer", "scope": "winter", "intensity": "noticeably"},
          {"axis": "less_rainy", "scope": "winter", "intensity": "noticeably"}]),
        ("cold snowy winters",
         [{"axis": "colder", "scope": "winter", "intensity": "noticeably"},
          {"axis": "wetter", "scope": "winter", "intensity": "noticeably"}]),
    ]

    print("=" * 80)
    print(f"{'query':<38} {'classical':>10} {'contrastive':>13} {'overlap':>9}")
    print("=" * 80)
    rows = []
    for text, vibes in cases:
        cls_top, exp_d, obs_d_cls, touched = classical_no_anchor_topk(
            profiles_scaled, vibes, top_k=10
        )
        ctr_top = contrastive_topk(text, st_model, text_proj, city_embs, top_k=10)
        _, _, obs_d_ctr = (
            *score_external_topk(ctr_top, profiles_scaled, vibes)[:1],
            score_external_topk(ctr_top, profiles_scaled, vibes)[1],
            score_external_topk(ctr_top, profiles_scaled, vibes)[1],
        )[:3]
        # cleaner: recompute
        exp_d2, obs_d_ctr, touched2 = score_external_topk(ctr_top, profiles_scaled, vibes)

        cls_cos, _, cls_obs_norm = sigma_score(obs_d_cls, exp_d)
        ctr_cos, _, ctr_obs_norm = sigma_score(obs_d_ctr, exp_d2)
        overlap = len(set(cls_top) & set(ctr_top))
        rows.append({
            "text": text, "vibes": vibes,
            "cls_top": cls_top, "ctr_top": ctr_top,
            "cls_cos": cls_cos, "ctr_cos": ctr_cos,
            "cls_obs_norm": cls_obs_norm, "ctr_obs_norm": ctr_obs_norm,
            "exp_d": exp_d, "obs_d_cls": obs_d_cls, "obs_d_ctr": obs_d_ctr,
            "touched": touched,
        })
        print(f"{text:<38} {cls_cos:+10.3f} {ctr_cos:+13.3f} {overlap:>9}/10")

    print()
    cls_arr = np.array([r["cls_cos"] for r in rows])
    ctr_arr = np.array([r["ctr_cos"] for r in rows])
    print(f"summary: classical mean cos = {cls_arr.mean():+.3f}, "
          f"contrastive mean cos = {ctr_arr.mean():+.3f}")
    print(f"         classical wins: "
          f"{(cls_arr > ctr_arr).sum()}/{len(rows)}")
    print()

    for r in rows:
        print("=" * 80)
        print(f"\"{r['text']}\"   "
              f"vibes = {[(v['axis'], v['scope']) for v in r['vibes']]}")
        print(f"  expected: {summarize_delta(r['exp_d'], r['touched'])}   "
              f"|exp|={np.linalg.norm(r['exp_d']):.2f}sigma")
        print(f"  CLASSICAL    cos={r['cls_cos']:+.3f}   "
              f"|obs|={r['cls_obs_norm']:.2f}sigma")
        print(f"    observed: {summarize_delta(r['obs_d_cls'], r['touched'])}")
        print(f"    top-10:   {[CITY_NAMES[i] for i in r['cls_top']]}")
        print(f"  CONTRASTIVE  cos={r['ctr_cos']:+.3f}   "
              f"|obs|={r['ctr_obs_norm']:.2f}sigma")
        print(f"    observed: {summarize_delta(r['obs_d_ctr'], r['touched'])}")
        print(f"    top-10:   {[CITY_NAMES[i] for i in r['ctr_top']]}")
        print()


if __name__ == "__main__":
    main()
