from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..services.contrastive_manifest import load_manifest

router = APIRouter(prefix="/contrastive", tags=["contrastive"])


@router.get("/manifest")
def get_manifest():
    """Static manifest of trained Stage 1 weights + city embeddings.

    The frontend fetches this once, caches it, and runs MiniLM + the
    projection head + cosine ranking entirely in the browser. Payload is
    immutable per training run, so we let the client cache aggressively.
    """
    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return JSONResponse(
        content=manifest,
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
