from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import tournament as svc

router = APIRouter(prefix="/tournament", tags=["tournament"])


class HistoryItem(BaseModel):
    shown: list[str]
    picked: str


class PairIn(BaseModel):
    history: list[HistoryItem] = []
    seed: int | None = None


class CityStatsOut(BaseModel):
    temp_c: float
    humidity_pct: float
    precip_mm: float
    sun_pct: float


class PairCityOut(BaseModel):
    name: str
    country: str
    lat: float
    lon: float
    image_url: str
    thumb_url: str
    stats: CityStatsOut


class PairOut(BaseModel):
    round: int
    total_rounds: int
    pair: list[PairCityOut]


class ReasonOut(BaseModel):
    label: str
    detail: str
    matched: bool


class CityResultOut(BaseModel):
    city: str
    country: str
    lat: float
    lon: float
    similarity: float
    reasons: list[ReasonOut]


class FinalOut(BaseModel):
    rounds_completed: int
    picked: list[str]
    results: list[CityResultOut]


@router.post("/pair", response_model=PairOut)
def get_pair(body: PairIn):
    try:
        result = svc.next_pair(
            [h.model_dump() for h in body.history],
            seed=body.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PairOut(
        round=result.round,
        total_rounds=result.total_rounds,
        pair=[
            PairCityOut(
                name=c.name, country=c.country, lat=c.lat, lon=c.lon,
                image_url=c.image_url, thumb_url=c.thumb_url,
                stats=CityStatsOut(**c.stats.__dict__),
            )
            for c in result.pair
        ],
    )


@router.post("/final", response_model=FinalOut)
def get_final(body: PairIn):
    result = svc.finalize([h.model_dump() for h in body.history])
    return FinalOut(
        rounds_completed=result.rounds_completed,
        picked=result.picked,
        results=[
            CityResultOut(
                city=r.city, country=r.country, lat=r.lat, lon=r.lon,
                similarity=r.similarity,
                reasons=[ReasonOut(**re.__dict__) for re in r.reasons],
            )
            for r in result.results
        ],
    )
