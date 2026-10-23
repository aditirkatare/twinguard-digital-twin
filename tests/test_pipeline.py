import pandas as pd

from src.data_generator import generate_synthetic_data
from src.preprocessing import make_forecasting_table
from src.reliability import assess_reliability


def test_generator_is_reproducible():
    a = generate_synthetic_data(n_patients=2, days=2, seed=7)
    b = generate_synthetic_data(n_patients=2, days=2, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_forecasting_target_is_future_reading():
    df = pd.DataFrame({
        "patient_id": ["A"] * 20,
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="5min"),
        "glucose_mg_dl": list(range(100, 120)),
        "heart_rate_bpm": [70.0] * 20,
        "activity_level": [0.2] * 20,
    })
    table = make_forecasting_table(df, horizon_steps=2)
    row = table.iloc[0]
    assert row["target_glucose"] == row["glucose_lag_0"] + 2


def test_reliability_abstains_when_data_is_missing():
    df = pd.DataFrame({"glucose_mg_dl": [100.0, None, None, 110.0]})
    result = assess_reliability(df, missing_threshold=0.2, min_readings=3)
    assert result["abstain"] is True
    assert result["status"] == "ABSTAIN"


def test_prediction_abstains_when_recent_glucose_is_missing():
    from src.prediction import predict_with_abstention

    data = generate_synthetic_data(
        n_patients=3, days=7, seed=42, missing_rate=0.0
    )
    history = data[data["patient_id"] == data["patient_id"].iloc[0]].copy()

    # Force the most recent readings to be missing.
    history = history.sort_values("timestamp")
    history.loc[history.tail(12).index, "glucose_mg_dl"] = None

    result = predict_with_abstention(data, history)

    assert result["status"] == "ABSTAIN"
    assert result["forecast"] is None
    assert result["reasons"]


def test_prediction_interval_is_ordered_when_prediction_allowed():
    from src.prediction import predict_with_abstention

    data = generate_synthetic_data(
        n_patients=3, days=7, seed=42, missing_rate=0.0
    )
    patient = data["patient_id"].iloc[0]
    history = data[data["patient_id"] == patient].copy().sort_values("timestamp")

    result = predict_with_abstention(
        data, history, max_interval_width=1000.0
    )

    if result["status"] == "PREDICTION_ALLOWED":
        assert result["lower_bound"] <= result["forecast"]
        assert result["forecast"] <= result["upper_bound"]
    else:
        assert result["forecast"] is None
    
    
def test_preprocessing_retains_rows_with_missing_wearable_data():
    from src.data_generator import generate_synthetic_data
    from src.preprocessing import make_forecasting_table

    data = generate_synthetic_data(
        n_patients=3,
        days=3,
        seed=42,
        missing_rate=0.10,
    )

    table = make_forecasting_table(data, horizon_steps=6)

    assert len(table) > 0
    assert table["target_glucose"].notna().all()
    assert table["target_timestamp"].notna().all()

    # Missing wearable features should no longer automatically
    # cause rows to be removed from the forecasting table.
    assert table["heart_rate_lag_0"].isna().any() or table["activity_lag_0"].isna().any()

    
def test_prediction_abstains_when_recent_glucose_is_insufficient():
    from src.data_generator import generate_synthetic_data
    from src.prediction import predict_with_abstention

    data = generate_synthetic_data(
        n_patients=3,
        days=3,
        seed=42,
        missing_rate=0.0,
    )

    history = data[data["patient_id"] == "SYN-001"].copy()
    history.loc[history.index[-12:], "glucose_mg_dl"] = float("nan")

    result = predict_with_abstention(data=data, history=history)

    assert result["status"] == "ABSTAIN"
    assert result["forecast"] is None
    assert "insufficient_recent_glucose_readings" in result["reasons"]

    
def test_prediction_can_proceed_with_missing_wearable_data():
    from src.data_generator import generate_synthetic_data
    from src.prediction import predict_with_abstention

    data = generate_synthetic_data(
        n_patients=8,
        days=7,
        seed=42,
        missing_rate=0.0,
    )

    # Make wearable inputs unavailable while preserving glucose history.
    data["heart_rate_bpm"] = float("nan")
    data["activity_level"] = float("nan")

    history = data[data["patient_id"] == "SYN-001"].copy()

    result = predict_with_abstention(
        data=data,
        history=history,
        max_interval_width=120.0,
    )

    # The prototype should not reject the forecast solely because
    # wearable features are missing.
    assert result["status"] == "PREDICTION_ALLOWED", result["reasons"]
    assert result["forecast"] is not None
    assert result["lower_bound"] <= result["forecast"] <= result["upper_bound"]



