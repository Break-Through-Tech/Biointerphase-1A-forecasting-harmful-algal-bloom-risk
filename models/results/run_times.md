# Run times

How long each step of `run_all.sh` took (wall-clock time). The weekly table step rebuilds
the table from the raw data; each model step cross-validates that model on its feature sets
with both schemes and saves the results.

| step | 1 week ahead | 2 weeks ahead | 4 weeks ahead | all horizons |
| --- | --- | --- | --- | --- |
| weekly_table | 8 s | 24 s | 12 s | 44 s |
| persistence | 4 s | 9 s | 3 s | 16 s |
| seasonal_climatology | 4 s | 10 s | 3 s | 17 s |
| logistic_regression | 10 s | 26 s | 11 s | 47 s |
| random_forest | 3 min 0 s | 6 min 43 s | 3 min 0 s | 12 min 43 s |
| xgboost | 2 min 23 s | 3 min 51 s | 2 min 8 s | 8 min 22 s |
| hybrid_escalation | 29 min 7 s | 27 min 41 s | 40 min 41 s | 97 min 29 s |
| bootstrap | 28 min 29 s | 6 min 42 s | 16 min 30 s | 51 min 41 s |
| total | 63 min 25 s | 46 min 6 s | 62 min 48 s | 172 min 19 s |

Measured on:
- Machine: Windows 11, Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 14 logical CPUs
- Python 3.14.2
- pandas 3.0.6
- numpy 2.5.4
- scikit-learn 1.9.1
- xgboost 3.4.1
