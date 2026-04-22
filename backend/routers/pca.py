from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services import pca_service
from ..services.nl_parser import ParsedQuery

router = APIRouter(prefix="/pca", tags=["pca"])


class CityPointOut(BaseModel):
    name: str
    country: str
    lat: float
    lon: float
    pc1: float
    pc2: float
    pc3: float


class LoadingOut(BaseModel):
    label: str
    feature: str
    month: str
    weight: float


class PCAOverviewOut(BaseModel):
    cities: list[CityPointOut]
    pc1_top: list[LoadingOut]
    pc2_top: list[LoadingOut]
    pc3_top: list[LoadingOut]
    pc1_label: str
    pc2_label: str
    pc3_label: str
    explained_variance: list[float]


class ProjectRequest(BaseModel):
    text: str


class ProjectOut(BaseModel):
    pc1: float
    pc2: float
    pc3: float
    anchor_name: str | None
    parsed: ParsedQuery


@router.get("", response_model=PCAOverviewOut)
def overview():
    o = pca_service.get_overview()
    return PCAOverviewOut(
        cities=[CityPointOut(**c.__dict__) for c in o.cities],
        pc1_top=[LoadingOut(**l.__dict__) for l in o.pc1_top],
        pc2_top=[LoadingOut(**l.__dict__) for l in o.pc2_top],
        pc3_top=[LoadingOut(**l.__dict__) for l in o.pc3_top],
        pc1_label=o.pc1_label,
        pc2_label=o.pc2_label,
        pc3_label=o.pc3_label,
        explained_variance=o.explained_variance,
    )


@router.post("/project", response_model=ProjectOut)
def project(req: ProjectRequest, db: Session = Depends(get_db)):
    p = pca_service.project_text(db, req.text)
    return ProjectOut(
        pc1=p.pc1, pc2=p.pc2, pc3=p.pc3,
        anchor_name=p.anchor_name, parsed=p.parsed,
    )
