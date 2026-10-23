# Evaluation protocol

The prototype uses a chronological split within each synthetic participant.
The final 20% of each participant's usable timeline is held out from training.

Report MAE and RMSE for the learned model and persistence baseline. Also report
the number of test observations. Because the data are synthetic, these metrics
only test the software pipeline and do not demonstrate clinical performance.

Next evaluation additions: residual-based prediction intervals, empirical
coverage and interval width, and controlled missingness stress tests.
