from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Float, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


class Vote(Base):
    __tablename__ = "votes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    city_a: Mapped[str] = mapped_column(String(128))
    city_b: Mapped[str] = mapped_column(String(128))
    winner: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FetchedCity(Base):
    """Cities not in the canonical 102 list, fetched from Open-Meteo on-demand."""
    __tablename__ = "fetched_cities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), unique=True, index=True)  # normalized lookup key
    name: Mapped[str] = mapped_column(String(128))  # canonical name from geocoder
    country: Mapped[str] = mapped_column(String(128), default="")
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    profile: Mapped[bytes] = mapped_column(LargeBinary)  # np.float32 96-d, .tobytes()
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
