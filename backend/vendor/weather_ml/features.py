"""Vendored subset of weather-ml/features.py.

Only the profile-building path is kept — the sequence/forecasting code and
its torch dependency are removed, since weather-ml-v2 never trains a model.
"""
import numpy as np
import pandas as pd  # noqa: F401 - kept for consistency with upstream

from data import weather_class  # resolved via sys.path shim in core.weather_ml

CONTINUOUS_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "cloud_cover", "pressure_msl", "wind_speed_10m"
]


def clean_df(df):
    """Fill missing values and clip outliers."""
    df = df.copy()
    for col in CONTINUOUS_VARS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    if "weather_code" in df.columns:
        df["weather_code"] = df["weather_code"].fillna(0).astype(int)
    return df


def make_city_profile(df):
    """Aggregate historical data into a 96-dim city weather profile.

    12 months x 8 variables (7 continuous means + dominant weather class ratio).
    Works with both hourly data (preferred) and daily data (fallback).
    """
    df = clean_df(df)
    df = df.copy()
    df["month"] = df.index.month

    col_map = {}
    for col in CONTINUOUS_VARS:
        if col in df.columns:
            col_map[col] = col
    if "temperature_2m" not in df.columns and "temperature_2m_max" in df.columns:
        df["temperature_2m"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        col_map["temperature_2m"] = "temperature_2m"
    if "precipitation" not in df.columns and "precipitation_sum" in df.columns:
        col_map["precipitation"] = "precipitation_sum"
    if "wind_speed_10m" not in df.columns and "wind_speed_10m_max" in df.columns:
        col_map["wind_speed_10m"] = "wind_speed_10m_max"

    profile = []
    for month in range(1, 13):
        month_data = df[df["month"] == month]
        if len(month_data) == 0:
            profile.extend([0.0] * 8)
            continue

        for col in CONTINUOUS_VARS:
            mapped = col_map.get(col)
            if mapped and mapped in month_data.columns:
                profile.append(float(month_data[mapped].mean()))
            else:
                profile.append(0.0)

        if "weather_code" in month_data.columns:
            clear_frac = (month_data["weather_code"].apply(weather_class) == 0).mean()
            profile.append(float(clear_frac))
        else:
            profile.append(0.5)

    return np.array(profile, dtype=np.float32)
