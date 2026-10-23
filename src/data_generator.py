"""Reproducible synthetic time-series for software testing, not clinical use."""

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_patients: int = 8,
    days: int = 7,
    seed: int = 42,
    missing_rate: float = 0.05,
) -> pd.DataFrame:
    """Generate five-minute glucose, heart-rate and activity observations."""
    if n_patients < 1 or days < 2:
        raise ValueError("n_patients must be positive and days must be at least 2.")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in [0, 1).")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        "2026-01-01", periods=days * 24 * 12, freq="5min"
    )
    frames = []

    for patient_num in range(n_patients):
        n = len(timestamps)
        t_hours = np.arange(n) / 12.0
        phase = rng.uniform(0, 2 * np.pi)
        baseline = rng.uniform(105, 145)

        daily = 12 * np.sin(2 * np.pi * t_hours / 24 + phase)
        slower = 5 * np.sin(2 * np.pi * t_hours / 8 + phase / 2)
        noise = rng.normal(0, 4, n)

        glucose = baseline + daily + slower + noise
        # Add illustrative meal-like excursions, not a physiological simulator.
        for hour in (8, 13, 19):
            for day in range(days):
                center = day * 24 + hour + rng.normal(0, 0.35)
                glucose += 22 * np.exp(-0.5 * ((t_hours - center) / 0.7) ** 2)

        glucose = np.clip(glucose, 55, 260)
        heart_rate = 72 + 7 * np.sin(2 * np.pi * t_hours / 24) + rng.normal(0, 3, n)
        activity = np.clip(
            0.25 + 0.2 * np.sin(2 * np.pi * (t_hours - 7) / 24)
            + rng.normal(0, 0.12, n),
            0,
            1,
        )

        # Simulate missing readings in short blocks.
        glucose_mask = rng.random(n) < missing_rate
        wearable_mask = rng.random(n) < min(missing_rate * 1.5, 0.8)
        for mask in (glucose_mask, wearable_mask):
            starts = np.where(rng.random(n) < missing_rate / 8)[0]
            for start in starts:
                mask[start:min(start + int(rng.integers(2, 7)), n)] = True

        glucose[glucose_mask] = np.nan
        heart_rate[wearable_mask] = np.nan
        activity[wearable_mask] = np.nan

        frames.append(pd.DataFrame({
            "patient_id": f"SYN-{patient_num + 1:03d}",
            "timestamp": timestamps,
            "glucose_mg_dl": glucose,
            "heart_rate_bpm": heart_rate,
            "activity_level": activity,
            "is_synthetic": True,
        }))

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_glucose.csv", index=False)
    print(f"Generated {len(df):,} synthetic rows.")
    print("Saved to data/synthetic_glucose.csv")
