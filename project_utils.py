from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "Demand"
GROUP_COLUMNS = ["Store ID", "Product ID"]

CATEGORICAL_FEATURES = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]

NUMERIC_FEATURES = [
    "Price",
    "Discount",
    "Competitor Pricing",
    "Promotion",
    "Epidemic",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "day_of_year",
    "is_weekend",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    "demand_roll_mean_7",
    "demand_roll_mean_28",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features using the Date column."""
    result = data.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    result["day_of_week"] = result["Date"].dt.dayofweek
    result["day_of_month"] = result["Date"].dt.day
    result["month"] = result["Date"].dt.month
    result["quarter"] = result["Date"].dt.quarter
    result["day_of_year"] = result["Date"].dt.dayofyear
    result["is_weekend"] = (result["Date"].dt.dayofweek >= 5).astype(int)
    return result


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag/rolling features based only on earlier Demand observations
    for each Store ID + Product ID combination.
    """
    result = add_calendar_features(data)
    result = result.sort_values(GROUP_COLUMNS + ["Date"]).copy()

    grouped = result.groupby(GROUP_COLUMNS, sort=False)["Demand"]
    for lag in [1, 7, 14, 28]:
        result[f"demand_lag_{lag}"] = grouped.shift(lag)

    result["demand_roll_mean_7"] = result.groupby(GROUP_COLUMNS, sort=False)[
        "Demand"
    ].transform(lambda s: s.shift(1).rolling(window=7, min_periods=7).mean())

    result["demand_roll_mean_28"] = result.groupby(GROUP_COLUMNS, sort=False)[
        "Demand"
    ].transform(lambda s: s.shift(1).rolling(window=28, min_periods=28).mean())

    return result


def season_for_month(month: int) -> str:
    """Simple calendar season mapping used for future forecast assumptions."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def make_future_feature_row(
    forecast_date: pd.Timestamp,
    store_id: str,
    product_id: str,
    category: str,
    region: str,
    price: float,
    discount: float,
    competitor_pricing: float,
    promotion: int,
    weather_condition: str,
    epidemic: int,
    demand_history: list[float],
) -> dict:
    """Build one future row in the same feature format used during training."""
    date = pd.Timestamp(forecast_date)
    history = list(demand_history)

    if len(history) < 28:
        raise ValueError("At least 28 days of historical demand are required.")

    return {
        "Store ID": store_id,
        "Product ID": product_id,
        "Category": category,
        "Region": region,
        "Weather Condition": weather_condition,
        "Seasonality": season_for_month(date.month),
        "Price": float(price),
        "Discount": float(discount),
        "Competitor Pricing": float(competitor_pricing),
        "Promotion": int(promotion),
        "Epidemic": int(epidemic),
        "day_of_week": int(date.dayofweek),
        "day_of_month": int(date.day),
        "month": int(date.month),
        "quarter": int(date.quarter),
        "day_of_year": int(date.dayofyear),
        "is_weekend": int(date.dayofweek >= 5),
        "demand_lag_1": float(history[-1]),
        "demand_lag_7": float(history[-7]),
        "demand_lag_14": float(history[-14]),
        "demand_lag_28": float(history[-28]),
        "demand_roll_mean_7": float(np.mean(history[-7:])),
        "demand_roll_mean_28": float(np.mean(history[-28:])),
    }
