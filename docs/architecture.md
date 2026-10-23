# TwinGuard Architecture

```text
Synthetic or documented dataset
            |
            v
Timestamp validation and preprocessing
            |
            v
Lag / rolling-window feature engineering
            |
            v
Patient-specific recent-state summary
            |
            v
30-minute glucose forecast
            |
            v
Data reliability checks + uncertainty layer
            |
       +----+----+
       |         |
  Forecast    Abstain
       |         |
       +----+----+
            |
            v
      Streamlit dashboard
```

## Current scope
The initial implementation is a research proof of concept. Synthetic signals
are used for pipeline testing, not clinical validation. Reliability rules are
transparent heuristics, not clinically validated safety thresholds.
