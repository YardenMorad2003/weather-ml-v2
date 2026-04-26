from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.session import Base, engine
from .db import models  # noqa: F401 - register models
from .routers import recommend as recommend_router
from .routers import pca as pca_router
from .routers import cities as cities_router
from .routers import tournament as tournament_router
from .routers import contrastive as contrastive_router

app = FastAPI(title="Weather ML v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yardenmorad2003.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(recommend_router.router)
app.include_router(pca_router.router)
app.include_router(cities_router.router)
app.include_router(tournament_router.router)
app.include_router(contrastive_router.router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"ok": True}
