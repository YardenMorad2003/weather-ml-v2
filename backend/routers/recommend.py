from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services.recommender import recommend_from_text
from ..services.nl_parser import ParsedQuery

router = APIRouter(prefix="/recommend", tags=["recommend"])


class TextQuery(BaseModel):
    text: str
    top_k: int = 10


class AnchorOut(BaseModel):
    name: str
    country: str
    lat: float
    lon: float
    source: str


class ReasonOut(BaseModel):
    label: str
    detail: str
    matched: bool


class CityOut(BaseModel):
    city: str
    country: str
    lat: float
    lon: float
    similarity: float
    reasons: list[ReasonOut]


class AnchorErrorOut(BaseModel):
    input: str
    suggestions: list[str]


class SaturationOut(BaseModel):
    anchor: str
    axes: list[str]


class ConflictOut(BaseModel):
    pairs: list[dict]


class RecommendOut(BaseModel):
    parsed: ParsedQuery
    anchor: AnchorOut | None
    results: list[CityOut]
    anchor_error: AnchorErrorOut | None = None
    saturation: SaturationOut | None = None
    conflict: ConflictOut | None = None


@router.post("/text", response_model=RecommendOut)
def recommend_text(q: TextQuery, db: Session = Depends(get_db)):
    resp = recommend_from_text(db, q.text, top_k=q.top_k)
    return RecommendOut(
        parsed=resp.parsed,
        anchor=(
            AnchorOut(
                name=resp.anchor.name, country=resp.anchor.country,
                lat=resp.anchor.lat, lon=resp.anchor.lon, source=resp.anchor.source,
            ) if resp.anchor else None
        ),
        results=[
            CityOut(
                city=r.city, country=r.country, lat=r.lat, lon=r.lon,
                similarity=r.similarity,
                reasons=[ReasonOut(**re.__dict__) for re in r.reasons],
            )
            for r in resp.results
        ],
        anchor_error=(
            AnchorErrorOut(
                input=resp.anchor_error.input,
                suggestions=resp.anchor_error.suggestions,
            ) if resp.anchor_error else None
        ),
        saturation=(
            SaturationOut(
                anchor=resp.saturation.anchor,
                axes=resp.saturation.axes,
            ) if resp.saturation else None
        ),
        conflict=(
            ConflictOut(pairs=resp.conflict.pairs) if resp.conflict else None
        ),
    )
