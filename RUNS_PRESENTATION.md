# Runs Presentation

This is the short presentation version of the run review. It focuses only on the runs and messages that are useful on slides.

## Core Message

- The best result in the repo is the resumed custom `SAC + LSTM` line with anti-stall training.
- The first real breakthrough happened one stage earlier, when the same line first became positive on clean evaluation.
- Standard SB3 baselines were useful for showing task difficulty, but they did not solve hardcore.
- The most honest conclusion is: strong partial success, not a fully solved policy.

## Best Runs At A Glance

| Run | What It Shows | Main Result | Presentation Message |
| --- | --- | --- | --- |
| `U3: ppo_bipedalwalker_seed42` | SB3 PPO hardcore baseline | `-98.81 +/- 20.90` over 5 eps | Better than random, but still clearly unsuccessful. |
| `U4: td3_bipedalwalker_seed42` | SB3 TD3 hardcore baseline | `-116.88 +/- 24.80` over 5 eps, length `2000.0` | Survives longer, but mostly stalls instead of solving obstacles. |
| `U5: legacy_sac_lstm_h12_s42` | First useful custom recurrent improvement | Saved clean test `-31.64` | Custom recurrent control helped, but was still below positive clean performance. |
| `U7: fix_train_a001_as` | First positive breakthrough | Best periodic clean eval `67.62`; video-backed clean eval `40.01 +/- 67.98` over 20 eps | First run with clearly positive and repeatable hardcore progress. |
| `U8: res_train_a001_as` | Strongest line in the repo | Best periodic clean eval `239.39`; video-backed clean eval `160.90 +/- 113.97` over 5 eps | Near-solved behavior on some seeds, but still unstable across seeds. |

## Simple Storyline For Slides

1. Start with SB3 baselines.
   They establish that `BipedalWalkerHardcore-v3` is genuinely difficult.
2. Show the first custom recurrent SAC result.
   This demonstrates that sequence memory helped more than the vanilla baselines.
3. Introduce anti-stall as a training aid.
   This was the change that moved the project from "less bad" to clearly positive clean results.
4. Show the resumed anti-stall SAC-LSTM run.
   This is the strongest result and the main contribution to highlight.
5. End with variance.
   The policy is strong, but not yet consistently solved across seeds.

## Slide-Friendly Interpretation

### 1. Baselines

- PPO: weak improvement over random, but still failing early.
- TD3: longer survival, but mostly unproductive motion.
- Takeaway: hardcore cannot be solved reliably with the default baseline setup used here.

### 2. First Custom Improvement

- `legacy_sac_lstm_h12_s42` beat the SB3 baselines.
- It still stayed below positive clean reward.
- Takeaway: recurrence helped, but it was not enough on its own.

### 3. Breakthrough

- `fix_train_a001_as` was the first line with clearly positive clean evaluation.
- It produced useful episodes, but also many unstable ones.
- Takeaway: anti-stall training created the first real breakthrough.

### 4. Strongest Result

- `res_train_a001_as` repeatedly reached `200+` clean eval during training.
- The saved video-backed evaluation reached `160.90 +/- 113.97` over 5 episodes.
- Two saved episodes were near-solved, but one was still weak.
- Takeaway: best policy in the repo, but not yet fully reliable.

## Recommended Video Order

1. `artifacts/runs/standard/ppo_bipedalwalker_seed42/videos/`
   Use this to show baseline failure.
2. `artifacts/runs/hardcore/legacy_sac_lstm_h12_s42/videos/`
   Use this to show that the custom recurrent line improved behavior.
3. `artifacts/runs/hardcore/fix_eval_best_raw/videos/`
   Use the best episode as the first positive turning point.
4. `artifacts/runs/hardcore/res_eval_best_raw/videos/`
   Show one elite episode first, then a weaker one.

## Suggested Ranking Slide

| Rank | Run | Why It Matters |
| --- | --- | --- |
| 1 | `U8 res_train_a001_as` | Best overall line; repeated `200+` clean evals; strongest videos. |
| 2 | `U7 fix_train_a001_as` | First clearly positive clean results. |
| 3 | `U5 legacy_sac_lstm_h12_s42` | Important early custom improvement over baselines. |
| 4 | `U3 ppo_bipedalwalker_seed42` | Useful baseline showing hardcore difficulty. |
| 5 | `U4 td3_bipedalwalker_seed42` | Useful failure mode: survival without real progress. |

## Caveats To Say Out Loud

- The best result is strong partial success, not a final solved hardcore policy.
- Only clean evaluations should be compared directly.
- Some archived report bundles are presentation artifacts, not separate new policies.
- Video evidence matters because episode length alone can hide stalling.

## One-Sentence Summary

The best result in the repo is the resumed custom `SAC + LSTM` anti-stall training line, which reached repeated clean evals above `200` and produced near-solved videos, but still showed enough variance that the most accurate presentation claim is strong partial success rather than full solution.
