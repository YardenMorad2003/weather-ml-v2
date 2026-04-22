from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from ..core import weather_ml  # noqa: F401
from cities import CITIES  # type: ignore
from ..services.state import get_state

router = APIRouter(prefix="/cities", tags=["cities"])

# per-month feature indices within each 8-tuple
F_TEMP, F_HUM, F_DEW, F_PRECIP, F_CLOUD, F_PRESS, F_WIND, F_CLEAR = range(8)


class AnnualStats(BaseModel):
    temp_c: float
    humidity_pct: float
    precip_mm: float
    wind_kmh: float
    clear_sky_frac: float


class CityDetail(BaseModel):
    name: str
    country: str
    lat: float
    lon: float
    annual: AnnualStats


@router.get("/{name}", response_model=CityDetail)
def city_detail(name: str):
    """Look up a canonical city by name (case-insensitive)."""
    key = name.strip().lower()
    idx = next((i for i, c in enumerate(CITIES) if c["name"].lower() == key), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="city not found")

    st = get_state()
    profile = st["profiles"][idx].reshape(12, 8)
    annual = profile.mean(axis=0)

    return CityDetail(
        name=CITIES[idx]["name"],
        country=CITIES[idx].get("country", ""),
        lat=CITIES[idx]["lat"],
        lon=CITIES[idx]["lon"],
        annual=AnnualStats(
            temp_c=float(annual[F_TEMP]),
            humidity_pct=float(annual[F_HUM]),
            precip_mm=float(annual[F_PRECIP]),
            wind_kmh=float(annual[F_WIND]),
            clear_sky_frac=float(annual[F_CLEAR]),
        ),
    )
