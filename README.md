# Bipedal Walker v3

- `train_bipedal_walker.py` is the simple script that trains, saves, evaluates, and can record a video

## Setup

```powershell
./setup_env.ps1
```

## Real Training

```powershell
.\.venv\Scripts\python.exe train_bipedal_walker.py --algo ppo --timesteps 50000 --eval-episodes 5 
```

The script saves a model under `artifacts/models/`, evaluates it, and prints a JSON summary with:

- `saved_model_path`
- `eval_mean_reward`
- `eval_std_reward`
- `eval_rewards`
- `eval_episode_lengths`

To also record a video, add `--record-video`:

```powershell
.\.venv\Scripts\python.exe train_bipedal_walker.py --algo sac --timesteps 50000 --record-video --video-episodes 1
```

Videos are written under `artifacts/videos/`.

If you created the environment before this update, rerun `./setup_env.ps1` once so `moviepy` is installed for video recording.

## Notes

- The manual PPO/SAC/TD3 code is there to show the RL ideas.
- The Stable-Baselines3 path is the one used for real training.
