# SB3 PPO Hardcore Baseline

## Objective
Proveriti koliko daleko standardni SB3 PPO moze da dogura na Hardcore modu bez custom sekvencijalnih modela.

## Verdict
- Status: `partial_baseline`
- Zakljucak: Poboljsava random baseline, ali i dalje ne daje upotrebljivo hodanje na Hardcore zadatku.
- Zasto je bitno: Koristan kao referentni baseline i dokaz da sam standardni PPO nije dovoljan.

## Key Metrics
- Evaluated episodes: 5
- Mean raw reward: -98.81
- Std raw reward: 20.90
- Mean episode length: 337.6
- Best episode: reward=-72.29, seed=43, length=255
- Worst episode: reward=-121.59, seed=44, length=218

## Bundled Artifacts
- Policy files: 2
- Video files: 2
- Log files: 1
