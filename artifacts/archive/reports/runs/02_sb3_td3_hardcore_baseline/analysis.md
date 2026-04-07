# SB3 TD3 Hardcore Baseline

## Objective
Uporediti TD3 baseline sa PPO-om na istom Hardcore zadatku.

## Verdict
- Status: `failed_baseline`
- Zakljucak: Politika uglavnom prezivljava do time-limit-a bez korisnog napretka i ostaje duboko negativna.
- Zasto je bitno: Dobar negativan primer koji opravdava potrebu za custom portom i checkpoint evaluacijom.

## Key Metrics
- Evaluated episodes: 5
- Mean raw reward: -116.88
- Std raw reward: 24.80
- Mean episode length: 2000.0
- Best episode: reward=-96.92, seed=46, length=2000
- Worst episode: reward=-162.60, seed=44, length=2000

## Bundled Artifacts
- Policy files: 1
- Video files: 2
- Log files: 1
