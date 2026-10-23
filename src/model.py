"""Baseline forecasts and purged chronological evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import make_forecasting_table


def evaluate_models(df):
    horizon_steps = 6
    table = make_forecasting_table(df, horizon_steps=horizon_steps)

    feature_cols = [
        c for c in table.columns
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

    # Record the actual timestamp used as each future target.
    # The target timestamp is metadata, never a model feature.
    if "target_timestamp" not in table.columns:
        target_times = []
        for _, group in df.assign(
            timestamp=pd.to_datetime(df["timestamp"])
        ).sort_values(["patient_id", "timestamp"]).groupby("patient_id"):
            target_times.extend(
                group["timestamp"].shift(-horizon_steps).tolist()
            )

        # Prefer an explicit target timestamp from preprocessing.
        # Rebuild the table with that field if it is not already present.
        raise ValueError(
            "target_timestamp is missing. Update preprocessing.py "
            "to create it before running this evaluation."
        )

    train_parts, test_parts = [], []

    for _, group in table.groupby("patient_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        split = int(len(group) * 0.8)

        if split < 1 or split >= len(group):
            continue

        test = group.iloc[split:].copy()
        test_start = test["timestamp"].min()

        # Purge training rows whose future target reaches the test period.
        train = group.iloc[:split].copy()
        train = train[train["target_timestamp"] < test_start]

        if not train.empty and not test.empty:
            train_parts.append(train)
            test_parts.append(test)

    if not train_parts or not test_parts:
        raise ValueError(
            "Not enough usable data after chronological splitting "
            "and purging the forecast horizon."
        )

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    X_train = train[feature_cols]
    y_train = train["target_glucose"]
    X_test = test[feature_cols]
    y_test = test["target_glucose"]

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)

    # Persistence predicts that glucose stays at its current value.
    persistence_pred = test["glucose_lag_0"].to_numpy()

    def metrics(y_true, pred):
        return {
            "mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "n": int(len(y_true)),
        }

    return {
        "model": model,
        "features": feature_cols,
        "train": train,
        "test": test,
        "model_predictions": model_pred,
        "persistence_predictions": persistence_pred,
        "model_metrics": metrics(y_test, model_pred),
        "persistence_metrics": metrics(y_test, persistence_pred),
    }
