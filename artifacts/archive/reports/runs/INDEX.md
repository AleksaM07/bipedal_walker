# Report Artifact Bundle

Ovaj folder okuplja glavne eksperimente u kratkim, stabilnim i citljivim report paketima.

| Experiment | Status | Mean raw reward | Folder |
|---|---|---:|---|
| SB3 PPO Hardcore Baseline | `partial_baseline` | -98.81 | `artifacts\report_runs\01_sb3_ppo_hardcore_baseline` |
| SB3 TD3 Hardcore Baseline | `failed_baseline` | -116.88 | `artifacts\report_runs\02_sb3_td3_hardcore_baseline` |
| Custom SAC + LSTM Best Raw Checkpoint | `strong_partial` | -22.45 | `artifacts\report_runs\03_custom_sac_lstm_best_raw` |
| Custom SAC + LSTM Anti-Stall Transfer Check | `diagnostic_only` | -74.50 | `artifacts\report_runs\04_custom_sac_lstm_antistall_transfer` |

Svaki eksperiment sadrzi podfoldere `policy/`, `videos/`, `summary/` i fajl `analysis.md`.

Bundle se regenerise komandom:

```powershell
.\.venv\Scripts\python.exe scripts\curate_report_runs.py
```
