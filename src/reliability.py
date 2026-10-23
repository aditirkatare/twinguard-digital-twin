"""Simple transparent data-quality rules for a prototype."""

import pandas as pd


def assess_reliability(
    recent: pd.DataFrame,
    missing_threshold: float = 0.20,
    min_readings: int = 6,
) -> dict:
    """Assess recent glucose availability; this is not clinical safety logic."""
    if "glucose_mg_dl" not in recent.columns:
        raise ValueError("recent data must contain glucose_mg_dl")

    values = recent["glucose_mg_dl"]
    missing_fraction = float(values.isna().mean()) if len(values) else 1.0
    valid_count = int(values.notna().sum())

    reasons = []
    if valid_count < min_readings:
        reasons.append("insufficient_recent_glucose_readings")
    if missing_fraction > missing_threshold:
        reasons.append("glucose_missingness_above_threshold")

    return {
        "missing_fraction": missing_fraction,
        "valid_readings": valid_count,
        "abstain": bool(reasons),
        "reasons": reasons,
        "status": "ABSTAIN" if reasons else "PREDICTION_ALLOWED",
    }
