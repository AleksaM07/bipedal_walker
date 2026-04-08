# Custom SAC + LSTM Anti-Stall Transfer Check

## Objective
Proveriti da li anti-stall shaping samo popravlja trening dinamiku ili i realno transferuje na cist Hardcore env.

## Verdict
- Status: `diagnostic_only`
- Zakljucak: Anti-stall pomaze treningu, ali checkpoint sa ep400 slabo generalizuje kada se meri bez anti-stall pravila.
- Zasto je bitno: Koristan kao alat za razbijanje lokalnog minimuma, ali ne kao finalni kriterijum uspeha.

## Key Metrics
- Evaluated episodes: 20
- Mean raw reward: -74.50
- Std raw reward: 22.97
- Mean episode length: 553.1
- Mean shaped reward: -34.00
- Best episode: reward=-39.08, seed=700054, length=750
- Worst episode: reward=-111.77, seed=700044, length=288

## Training Eval Highlights
- Best periodic eval: episode 600 | raw_mean=-38.50 | shaped_mean=-14.50
- Latest periodic eval found in log: episode 600 | raw_mean=-38.50 | shaped_mean=-14.50

## Bundled Artifacts
- Policy files: 1
- Video files: 1
- Log files: 2
