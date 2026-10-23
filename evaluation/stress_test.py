"""Reproducible synthetic missingness stress test for TwinGuard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_data
from src.preprocessing import make_forecasting_table

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evaluation" / "results"


def run_stress_test():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for missing_rate in (0.0, 0.05, 0.10, 0.20, 0.30):
        data = generate_synthetic_data(
            n_patients=8,
            days=7,
            seed=42,
            missing_rate=missing_rate,
        )

        total = len(data)
        glucose_missing = float(data["glucose_mg_dl"].isna().mean())
        heart_rate_missing = float(data["heart_rate_bpm"].isna().mean())
        activity_missing = float(data["activity_level"].isna().mean())

        try:
            table = make_forecasting_table(data, horizon_steps=6)
            usable_rows = len(table)
            usable_fraction = usable_rows / total if total else 0.0

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

            complete_feature_rows = int(
                table[feature_cols].notna().all(axis=1).sum()
            )

            status = "OK"
            error = None
        except (ValueError, KeyError) as exc:
            usable_rows = 0
            usable_fraction = 0.0
            complete_feature_rows = 0
            status = "ERROR"
            error = str(exc)

        rows.append({
            "configured_missing_rate": missing_rate,
            "observed_glucose_missing_fraction": glucose_missing,
            "observed_heart_rate_missing_fraction": heart_rate_missing,
            "observed_activity_missing_fraction": activity_missing,
            "raw_rows": total,
            "forecast_table_rows": usable_rows,
            "usable_row_fraction": usable_fraction,
            "complete_feature_rows": complete_feature_rows,
            "status": status,
            "error": error,
            "seed": 42,
            "patients": 8,
            "days_per_patient": 7,
            "forecast_horizon_minutes": 30,
            "data_type": "synthetic",
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "missingness_stress_test.csv", index=False)

    summary = {
        "experiment": "TwinGuard synthetic missingness stress test",
        "purpose": (
            "Measure missing-data prevalence and retained forecasting rows; "
            "not clinical validation."
        ),
        "seed": 42,
        "patients": 8,
        "days_per_patient": 7,
        "horizon_minutes": 30,
        "results": rows,
    }

    with open(OUTPUT_DIR / "missingness_stress_test.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(results.to_string(index=False))
    print("\nSaved:")
    print("evaluation/results/missingness_stress_test.csv")
    print("evaluation/results/missingness_stress_test.json")


if __name__ == "__main__":
    run_stress_test()
