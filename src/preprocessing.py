"""Causal time-series feature engineering with explicit future targets."""

import numpy as np
import pandas as pd

GLUCOSE = "glucose_mg_dl"


def make_forecasting_table(
    df: pd.DataFrame, horizon_steps: int = 6
) -> pd.DataFrame:
    """Create a future glucose target, six five-minute steps by default."""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be at least 1.")

    required = {"patient_id", "timestamp", GLUCOSE}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["patient_id", "timestamp"])
    data = data.drop_duplicates(["patient_id", "timestamp"], keep="last")

    parts = []

    for _, group in data.groupby("patient_id", sort=False):
        g = group.copy()
        glucose = g[GLUCOSE]

        # Past-only glucose features.
        for lag in (0, 1, 2, 3, 6, 12):
            g[f"glucose_lag_{lag}"] = glucose.shift(lag)

        # Use a one-hour window of past readings only.
        # min_periods cannot exceed the available window size.
        past_glucose = glucose.shift(1)

        g["glucose_mean_1h"] = (
            past_glucose.rolling(window=12, min_periods=1).mean()
        )
        g["glucose_std_1h"] = (
            past_glucose.rolling(window=12, min_periods=2).std()
        )

        # Current sensor readings are available at forecast time.
        if "heart_rate_bpm" in g.columns:
            g["heart_rate_lag_0"] = g["heart_rate_bpm"]
        else:
            g["heart_rate_lag_0"] = np.nan

        if "activity_level" in g.columns:
            g["activity_lag_0"] = g["activity_level"]
        else:
            g["activity_lag_0"] = np.nan

        minute_of_day = (
            g["timestamp"].dt.hour * 60 + g["timestamp"].dt.minute
        )
        g["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
        g["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

        # Explicit future target and its timestamp for leakage-safe purging.
        g["target_glucose"] = glucose.shift(-horizon_steps)
        g["target_timestamp"] = g["timestamp"].shift(-horizon_steps)

        parts.append(g)

    result = pd.concat(parts, ignore_index=True)

    feature_cols = [
        c for c in result.columns
        if c.startswith("glucose_lag_")
        or c in {
            "glucose_mean_1h",
            "glucose_std_1h",
            "heart_rate_lag_0",
            "activity_lag_0",
            "hour_sin",
            "hour_cos",
        }
    ]

    
    # Keep rows with missing wearable readings so the model can
    # handle them explicitly. A missing future glucose target cannot
    # be used for supervised training or evaluation.
    result = result.dropna(
        subset=["target_glucose", "target_timestamp"]
    ).copy()

    return result

