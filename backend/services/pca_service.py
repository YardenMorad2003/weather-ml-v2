"""PCA-based 3D climate explorer.

Fits PCA(3) on the StandardScaler'd, phase-aligned city profiles. Returns 3D
coords per city plus per-PC loadings and LLM-generated labels so each axis
is interpretable without climatology background.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core import weather_ml  # noqa: F401
from ..core.config import settings
from cities import CITIES  # type: ignore

from .state import get_state
from .nl_parser import parse_query, ParsedQuery, _get_client
from .vibe_table import apply_vibes
from .city_resolver import resolve

FEATURE_NAMES = ["temp", "humidity", "dewpoint", "precip", "cloud", "pressure", "wind", "clear_sky"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class CityPoint:
    name: str
    country: str
    lat: float
    lon: float
    pc1: float
    pc2: float
    pc3: float


@dataclass
class Loading:
    label: str      # e.g. "temp · Jan"
    feature: str
    month: str
    weight: float


@dataclass
class PCAOverview:
    cities: list[CityPoint]
    pc1_top: list[Loading]
    pc2_top: list[Loading]
    pc3_top: list[Loading]
    pc1_label: str
    pc2_label: str
    pc3_label: str
    explained_variance: list[float]


def _top_loadings(components_row: np.ndarray, k: int = 8) -> list[Loading]:
    """Return the k (feature, month) dims with largest |weight| for one PC.

    Profile layout is (12, 8) = months x features, flattened. After phase
    alignment "Jan" means the city's coldest-season slot regardless of
    hemisphere, so the labels are semantic rather than calendar-literal.
    """
    order = np.argsort(-np.abs(components_row))[:k]
    out: list[Loading] = []
    for idx in order:
        month_i = int(idx // 8)
        feat_i = int(idx % 8)
        out.append(Loading(
            label=f"{FEATURE_NAMES[feat_i]} · {MONTHS[month_i]}",
            feature=FEATURE_NAMES[feat_i],
            month=MONTHS[month_i],
            weight=float(components_row[idx]),
        ))
    return out


class _AxisLabels(BaseModel):
    pc1: str
    pc2: str
    pc3: str


def _label_axes(
    pc1_top: list[Loading],
    pc2_top: list[Loading],
    pc3_top: list[Loading],
) -> tuple[str, str, str]:
    def fmt(loadings: list[Loading]) -> str:
        return ", ".join(
            f"{l.feature} in {l.month} ({'+' if l.weight >= 0 else ''}{l.weight:.2f})"
            for l in loadings
        )

    client = _get_client()
    completion = client.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You label PCA axes of a climate embedding for interpretability. "
                    "Each axis is a direction in 96-d climate space (12 months x 8 features: "
                    "temp, humidity, dewpoint, precip, cloud, pressure, wind, clear_sky). "
                    "Given the top (feature, month, signed weight) loadings, produce a "
                    "2-5 word label describing what a city HIGH on that axis feels like. "
                    "Examples: 'Summer heat & humidity', 'Mild overcast winters', "
                    "'Dry sunny year-round', 'Big seasonal temperature swing'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PC1 top loadings: {fmt(pc1_top)}\n"
                    f"PC2 top loadings: {fmt(pc2_top)}\n"
                    f"PC3 top loadings: {fmt(pc3_top)}"
                ),
            },
        ],
        response_format=_AxisLabels,
    )
    labels = completion.choices[0].message.parsed
    return labels.pc1, labels.pc2, labels.pc3


def get_overview() -> PCAOverview:
    st = get_state()
    pca = st["pca"]
    coords = st["profiles_proj"]

    cities = [
        CityPoint(
            name=CITIES[i]["name"],
            country=CITIES[i].get("country", ""),
            lat=CITIES[i]["lat"],
            lon=CITIES[i]["lon"],
            pc1=float(coords[i, 0]),
            pc2=float(coords[i, 1]),
            pc3=float(coords[i, 2]),
        )
        for i in range(len(CITIES))
    ]

    pc1_top = _top_loadings(pca.components_[0])
    pc2_top = _top_loadings(pca.components_[1])
    pc3_top = _top_loadings(pca.components_[2])

    if "axis_labels" not in st:
        st["axis_labels"] = _label_axes(pc1_top, pc2_top, pc3_top)
    pc1_label, pc2_label, pc3_label = st["axis_labels"]

    return PCAOverview(
        cities=cities,
        pc1_top=pc1_top,
        pc2_top=pc2_top,
        pc3_top=pc3_top,
        pc1_label=pc1_label,
        pc2_label=pc2_label,
        pc3_label=pc3_label,
        explained_variance=[float(v) for v in pca.explained_variance_ratio_],
    )


@dataclass
class ProjectedPoint:
    pc1: float
    pc2: float
    pc3: float
    anchor_name: Optional[str]
    parsed: ParsedQuery


def project_text(db: Session, text: str) -> ProjectedPoint:
    """Run the same parse + vibe pipeline as /recommend/text, then
    project the resulting user vector into the 3D PCA space."""
    parsed = parse_query(text)
    st = get_state()
    profiles = st["profiles"]
    scaler = st["scaler"]
    pca = st["pca"]

    anchor = None
    if parsed.anchor_city:
        anchor = resolve(db, parsed.anchor_city, profiles)

    if anchor is not None:
        user_raw = anchor.profile.reshape(1, -1)
    else:
        user_raw = profiles.mean(axis=0, keepdims=True)

    user_scaled = scaler.transform(user_raw)[0]
    vibes = [v.model_dump() for v in parsed.vibes]
    user_scaled, _ = apply_vibes(user_scaled, vibes)

    xyz = pca.transform(user_scaled.reshape(1, -1))[0]
    return ProjectedPoint(
        pc1=float(xyz[0]),
        pc2=float(xyz[1]),
        pc3=float(xyz[2]),
        anchor_name=anchor.name if anchor else None,
        parsed=parsed,
    )
