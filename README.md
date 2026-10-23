# TwinGuard: An Uncertainty-Aware Digital Twin for Glucose Forecasting

**Project:** TwinGuard  
**Project Lead:** Aditi Rajendra Katare  
**College:** [Add your official college name]  
**Status:** Prototype under development

## Problem Statement
Glucose forecasting systems can give misleading predictions when sensor readings are missing, stale, corrupted, or outside the conditions represented in training data. TwinGuard explores how a personalized digital twin can assess data reliability, forecast short-term glucose trends, communicate uncertainty, and abstain when evidence is insufficient.

## Healthcare Use Case
Exploratory short-horizon glucose forecasting and glycemic instability monitoring. The prototype is intended for research and demonstration, not diagnosis, treatment, or clinical decision-making.

## Proposed Solution
TwinGuard combines a patient-specific state, timestamped glucose time-series, forecasting, data-quality monitoring, uncertainty estimation, and a reliability-aware abstention mechanism. A Streamlit dashboard will display glucose history, forecasts, uncertainty information, and sensor-quality warnings.

## Technical Stack
- Python
- NumPy and pandas
- scikit-learn
- Streamlit
- Matplotlib
- pytest

## AI/ML Model and Framework
The initial benchmark will compare a persistence forecast against a lightweight regression model using chronological evaluation. Planned metrics include MAE and RMSE. Prediction intervals, empirical coverage, interval width, and abstention rates will be evaluated as the corresponding components are implemented.

No performance results are claimed until experiments have been run. Synthetic data, if used, will be explicitly labelled and will not be treated as clinical evidence.

## Data and Validation
The initial proof of concept may use reproducible synthetic glucose time-series to validate the software pipeline and simulate missing sensor readings. Any real dataset will be documented with its source, access conditions, population, and limitations. Temporal leakage will be avoided through chronological train/test separation.

## Architecture
Data ingestion → preprocessing and quality checks → feature engineering → patient-state update → forecasting → uncertainty and reliability checks → forecast or abstain → dashboard.

See `docs/architecture.pdf` for the architecture diagram once exported.

## Demo Video
**Pending:** Add the publicly accessible or unlisted YouTube link for the required 15–20 minute demonstration.

## Presentation
**Pending:** Add the exported project presentation PDF or PPT.

## License
MIT License. See `LICENSE`. Third-party datasets and assets remain subject to their own terms.

## Safety and Limitations
TwinGuard is a research prototype, not a validated medical device. It must not be used for diagnosis, treatment, insulin dosing, or emergency decisions. Synthetic experiments do not demonstrate clinical effectiveness or real-world generalization.

## Phase 1 Submission Checklist
- [ ] Correct team and college details
- [ ] Project title, problem statement, and healthcare use case
- [ ] Technical stack and AI/ML details
- [ ] 15–20 minute demo video link
- [ ] Open-source license
- [ ] Architecture diagram PDF/PPT
- [ ] Project presentation PDF/PPT
- [ ] All files and links publicly accessible