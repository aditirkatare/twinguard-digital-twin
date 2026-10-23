import matplotlib.pyplot as plt
import streamlit as st

from src.data_generator import generate_synthetic_data
from src.model import evaluate_models
from src.prediction import predict_with_abstention
from src.reliability import assess_reliability
from src.twin import update_patient_state

st.set_page_config(page_title="TwinGuard", page_icon="🛡️", layout="wide")
st.title("TwinGuard")
st.caption(
    "Uncertainty-aware glucose forecasting | Synthetic proof of concept"
)
st.warning(
    "Synthetic demonstration only. Not a medical device. Do not use for "
    "diagnosis, treatment, insulin dosing, or emergency decisions."
)

with st.sidebar:
    st.header("Demo configuration")
    patients = st.slider("Synthetic participants", 3, 12, 8)
    days = st.slider("Days per participant", 3, 14, 7)
    missing_rate = st.slider("Synthetic missingness", 0.0, 0.30, 0.05, 0.01)
    seed = st.number_input("Random seed", min_value=0, value=42)
    max_interval_width = st.slider(
        "Maximum prototype interval width (mg/dL)", 20, 120, 60, 5
    )

data = generate_synthetic_data(
    n_patients=patients,
    days=days,
    seed=int(seed),
    missing_rate=missing_rate,
)
patient = st.selectbox(
    "Choose synthetic participant", sorted(data["patient_id"].unique())
)
history = data[data["patient_id"] == patient].sort_values("timestamp").copy()

state = update_patient_state(history)
reliability = assess_reliability(history.tail(12))

c1, c2, c3 = st.columns(3)
c1.metric("Recent readings used", state["readings_used"])
c2.metric(
    "Recent mean glucose",
    "N/A"
    if state["recent_mean_glucose"] is None
    else f"{state['recent_mean_glucose']:.1f} mg/dL",
)
c3.metric("Data-quality decision", reliability["status"])

st.subheader("30-minute forecast")
with st.spinner("Checking reliability and calibrating the prototype forecast..."):
    prediction = predict_with_abstention(
        data=data,
        history=history,
        max_interval_width=float(max_interval_width),
    )

if prediction["status"] == "ABSTAIN":
    st.error("Forecast withheld by the prototype.")
    st.write("Reasons:", ", ".join(prediction["reasons"]))
    st.caption("No point forecast is displayed when the prediction is withheld.")
else:
    st.success("Prototype checks passed. Forecast is available.")
    p1, p2, p3 = st.columns(3)
    p1.metric("Forecast", f"{prediction['forecast']:.1f} mg/dL")
    p2.metric("Lower interval bound", f"{prediction['lower_bound']:.1f} mg/dL")
    p3.metric("Upper interval bound", f"{prediction['upper_bound']:.1f} mg/dL")

st.caption(
    f"Nominal interval level: {prediction['nominal_coverage']:.0%} | "
    f"Calibration residuals: {prediction['calibration_samples']}"
)
st.caption(
    "Intervals are empirical prototype estimates, not guaranteed coverage. "
    "Time-series dependence and distribution shifts can invalidate nominal coverage."
)

st.subheader("Synthetic glucose history")
fig, ax = plt.subplots()
ax.plot(
    history["timestamp"],
    history["glucose_mg_dl"],
    label="Synthetic glucose",
)
ax.set_ylabel("Glucose (mg/dL)")
ax.set_xlabel("Timestamp")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

if st.button("Run chronological model evaluation"):
    with st.spinner("Training and evaluating baseline models..."):
        results = evaluate_models(data)

    st.subheader("Held-out synthetic-data metrics")
    st.write("Learned model", results["model_metrics"])
    st.write("Persistence baseline", results["persistence_metrics"])
    st.info(
        "These results are from synthetic data and must not be interpreted "
        "as clinical effectiveness."
    )
