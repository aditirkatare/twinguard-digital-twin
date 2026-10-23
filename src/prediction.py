
"""Prototype forecast intervals and explicit prediction abstention."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.preprocessing import make_forecasting_table
from src.reliability import assess_reliability


FEATURES = [
    "glucose_lag_0",
    "glucose_lag_1",
    "glucose_lag_2",
    "glucose_lag_3",
    "glucose_lag_6",
    "glucose_lag_12",
    "glucose_mean_1h",
    "glucose_std_1h",
    "heart_rate_lag_0",
    "activity_lag_0",
    "hour_sin",
    "hour_cos",
]


def _latest_features(history):
    """Build a forecast-time feature row using past/current observations only."""
    h = history.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    h = (
        h.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    if h.empty:
        raise ValueError("No history is available.")

    g = h["glucose_mg_dl"]
    row = {}

    for lag in (0, 1, 2, 3, 6, 12):
        row[f"glucose_lag_{lag}"] = (
            float(g.iloc[-1 - lag])
            if len(g) > lag and pd.notna(g.iloc[-1 - lag])
            else np.nan
        )

    past_glucose = g.iloc[:-1].tail(12)

    row["glucose_mean_1h"] = (
        float(past_glucose.mean())
        if past_glucose.notna().sum() >= 6
        else np.nan
    )
    row["glucose_std_1h"] = (
        float(past_glucose.std())
        if past_glucose.notna().sum() >= 6
        else np.nan
    )

    for source, feature in [
        ("heart_rate_bpm", "heart_rate_lag_0"),
        ("activity_level", "activity_lag_0"),
    ]:
        row[feature] = (
            float(h[source].iloc[-1])
            if source in h and pd.notna(h[source].iloc[-1])
            else np.nan
        )

    latest_time = h["timestamp"].iloc[-1]
    minute = latest_time.hour * 60 + latest_time.minute
    row["hour_sin"] = np.sin(2 * np.pi * minute / 1440)
    row["hour_cos"] = np.cos(2 * np.pi * minute / 1440)

    return pd.DataFrame([row], columns=FEATURES)


def predict_with_abstention(
    data,
    history,
    alpha=0.10,
    max_interval_width=60.0,
    missing_threshold=0.20,
    min_readings=6,
):
    """Fit and calibrate a prototype, then forecast or explicitly abstain.

    Calibration is chronological and separate from model fitting.
    Time-series dependence means nominal interval coverage is not guaranteed.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    recent = history.sort_values("timestamp").tail(12)
    reliability = assess_reliability(
        recent,
        missing_threshold=missing_threshold,
        min_readings=min_readings,
    )

    def abstain(reasons, reliability_result=None):
        return {
            "status": "ABSTAIN",
            "forecast": None,
            "lower_bound": None,
            "upper_bound": None,
            "interval_width": None,
            "reasons": reasons,
            "reliability": (
                reliability_result
                if reliability_result is not None
                else reliability
            ),
            "calibration_samples": 0,
            "nominal_coverage": 1 - alpha,
        }

    if reliability["abstain"]:
        return abstain(reliability["reasons"])

    try:
        table = make_forecasting_table(data, horizon_steps=6)

        train_parts = []
        calibration_parts = []

        for _, group in table.groupby("patient_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            n = len(group)

            train_end = int(n * 0.60)
            calibration_end = int(n * 0.80)

            if (
                train_end < 1
                or calibration_end <= train_end
                or calibration_end >= n
            ):
                continue

            train = group.iloc[:train_end].copy()
            calibration = group.iloc[train_end:calibration_end].copy()

            calibration_start = calibration["timestamp"].min()
            train = train[
                train["target_timestamp"] < calibration_start
            ]

            test_start = group.iloc[calibration_end]["timestamp"]
            calibration = calibration[
                calibration["target_timestamp"] < test_start
            ]

            if not train.empty and not calibration.empty:
                train_parts.append(train)
                calibration_parts.append(calibration)

        if not train_parts or not calibration_parts:
            return abstain(["insufficient_training_or_calibration_data"])

        train = pd.concat(train_parts, ignore_index=True)
        calibration = pd.concat(calibration_parts, ignore_index=True)

        # Remove features that have no observed training values.
        # The same feature set must be used in every prediction stage.
        usable_features = [
            feature
            for feature in FEATURES
            if train[feature].notna().any()
        ]

        if not usable_features:
            return abstain(["no_observed_training_features"])

        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )

        model.fit(
            train[usable_features],
            train["target_glucose"],
        )

        # Calibration must use the same features as training.
        calibration_pred = model.predict(
            calibration[usable_features]
        )

        residuals = np.abs(
            calibration["target_glucose"].to_numpy()
            - calibration_pred
        )

        if len(residuals) < 20:
            return abstain(["insufficient_calibration_residuals"])

        # Finite-sample split-conformal quantile convention.
        ordered = np.sort(residuals)
        rank = min(
            len(ordered),
            int(np.ceil((len(ordered) + 1) * (1 - alpha))),
        )
        q = float(ordered[rank - 1])

        latest = _latest_features(history)

        # Missing wearable features are acceptable when those features
        # were absent from training and therefore excluded. Missing
        # required forecast features still cause explicit abstention.
        if latest[usable_features].isna().any().any():
            return abstain(["latest_forecast_features_incomplete"])

        forecast = float(
            model.predict(latest[usable_features])[0]
        )

        lower = forecast - q
        upper = forecast + q
        width = upper - lower

        if width > max_interval_width:
            result = abstain(["prediction_interval_too_wide"])
            result["interval_width"] = float(width)
            result["calibration_samples"] = int(len(residuals))
            return result

        return {
            "status": "PREDICTION_ALLOWED",
            "forecast": forecast,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "interval_width": float(width),
            "reasons": [],
            "reliability": reliability,
            "calibration_samples": int(len(residuals)),
            "nominal_coverage": 1 - alpha,
        }

    except (ValueError, KeyError, IndexError) as exc:
        return abstain([f"prediction_pipeline_error: {exc}"])
