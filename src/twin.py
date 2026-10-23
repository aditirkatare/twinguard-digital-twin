"""Patient-specific state summary derived from observed history."""

import pandas as pd


def update_patient_state(history: pd.DataFrame) -> dict:
    """Summarize recent observations; does not model full physiology."""
    valid = history.dropna(subset=["glucose_mg_dl"]).sort_values("timestamp")
    recent = valid.tail(12)

    if recent.empty:
        return {
            "patient_id": str(history["patient_id"].iloc[-1]) if len(history) else "unknown",
            "readings_used": 0,
            "recent_mean_glucose": None,
            "recent_trend_mg_dl_per_5min": None,
        }

    trend = None
    if len(recent) >= 2:
        trend = float(
            (recent["glucose_mg_dl"].iloc[-1] - recent["glucose_mg_dl"].iloc[0])
            / (len(recent) - 1)
        )

    return {
        "patient_id": str(recent["patient_id"].iloc[-1]),
        "readings_used": int(len(recent)),
        "recent_mean_glucose": float(recent["glucose_mg_dl"].mean()),
        "recent_trend_mg_dl_per_5min": trend,
    }
