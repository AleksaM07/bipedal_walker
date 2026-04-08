# Custom SAC + LSTM Best Raw Checkpoint

## Objective
Izmeriti najbolji vanilla custom SAC+LSTM checkpoint na cistom Hardcore env-u.

## Verdict
- Status: `strong_partial`
- Zakljucak: Najjaci rezultat u trenutnom repou: veoma stabilan checkpoint koji ne pada, ali i dalje ostaje ispod nule.
- Zasto je bitno: Custom SAC + LSTM sa history ulazom pravi veliki pomak u odnosu na legacy baseline i predstavlja glavni kandidat za dalji rad.

## Key Metrics
- Evaluated episodes: 20
- Mean raw reward: -22.45
- Std raw reward: 3.98
- Mean episode length: 750.0
- Mean shaped reward: -22.45
- Best episode: reward=-20.23, seed=700049, length=750
- Worst episode: reward=-31.88, seed=700051, length=750

## Training Eval Highlights
- Best periodic eval: episode 600 | raw_mean=-21.58 | shaped_mean=-21.58
- Latest periodic eval found in log: episode 600 | raw_mean=-21.58 | shaped_mean=-21.58

## Bundled Artifacts
- Policy files: 1
- Video files: 1
- Log files: 2
