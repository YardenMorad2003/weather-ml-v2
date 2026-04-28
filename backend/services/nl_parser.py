"""Parse a free-text user query into an anchor city + vibes list.

Uses OpenAI structured outputs so the model is constrained to the fixed
vibe vocabulary and cannot hallucinate feature names or magnitudes.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

from ..core.config import settings
from .vibe_table import VIBE_AXES, SCOPES, INTENSITIES

# Keep these Literal types in sync with VIBE_AXES / SCOPES / INTENSITIES in
# vibe_table.py — they constrain OpenAI structured output to our vocab so the
# downstream dict lookups can't KeyError on a hallucinated axis like
# "less_humid" (the correct axes are "drier" / "less_muggy").
_VibeAxis = Literal[
    "warmer", "colder", "milder", "more_extreme",
    "more_seasonal", "less_seasonal",
    "drier", "more_humid", "less_muggy", "wetter", "less_rainy",
    "sunnier", "cloudier",
    "windier", "calmer",
]
_Scope = Literal["winter", "summer", "year_round"]
_Intensity = Literal["slightly", "noticeably", "much"]

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


class Vibe(BaseModel):
    axis: _VibeAxis
    scope: _Scope
    intensity: _Intensity


class ParsedQuery(BaseModel):
    anchor_city: Optional[str] = Field(
        default=None,
        description="City name the user referenced as a starting point, if any",
    )
    anchor_is_fictional: bool = Field(
        default=False,
        description=(
            "True ONLY if anchor_city is a mythological, fictional, or imaginary "
            "place — e.g. Atlantis, Hogwarts, Camelot, Mordor, Westeros, Wakanda, "
            "Olympus, Avalon, Shangri-La, Narnia, El Dorado, Eden, Gotham, "
            "Metropolis (the comic city), Pandora. False for any real-world city, "
            "including obscure ones, even if a famous fictional place shares the "
            "name. Set this based on the most famous/canonical referent of the "
            "name, not whether some real obscure place happens to share it."
        ),
    )
    vibes: list[Vibe] = Field(default_factory=list)


SYSTEM_PROMPT = f"""You translate short weather/climate wishes into a structured query.

Extract:
1. anchor_city: if the user names a city as a starting point, return it
   (return the name even if it's fictional — the next field marks it).
   Otherwise null.
2. anchor_is_fictional: True if the anchor is a mythological, fictional, or
   imaginary place (Atlantis, Hogwarts, Camelot, Mordor, Westeros, Wakanda,
   Olympus, Avalon, Shangri-La, Narnia, El Dorado, Eden, Gotham, Metropolis-
   the-comic-city, Pandora, etc.). False for any real-world city including
   obscure ones. Decide based on the most famous referent of the name.
3. vibes: a list of climate modifications.

Each vibe must use ONLY these values:
- axis: one of {VIBE_AXES}
- scope: one of {SCOPES}
- intensity: one of {INTENSITIES}

Rules:
- If the user says "like X but Y", anchor_city = X and vibes describe Y.
- If there is no anchor city, still return vibes describing the desired climate.
- Pick the closest axis. Do not invent new ones.
- Default scope to year_round unless the user mentions a season.
- Default intensity to noticeably unless the user uses words like "slightly",
  "a bit" (slightly) or "much", "way", "a lot" (much).
- Return an empty vibes list only if the user gives no climate signal at all.
"""


def parse_query(text: str) -> ParsedQuery:
    client = _get_client()
    completion = client.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format=ParsedQuery,
    )
    return completion.choices[0].message.parsed
