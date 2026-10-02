from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import load_config
from .data import fetch_klines
from .env import ReturnPredictionEnv
from .features import FEATURE_COLUMNS, build_features, fit_normalizer, normalized_values


def make_env(frame, mean, std, asset_index, asset_count, cfg):
    def factory():
        return ReturnPredictionEnv(
            normalized_values(frame, mean, std),
            frame["close"].to_numpy(),
            asset_index=asset_index,
            asset_count=asset_count,
            lookback=int(cfg["lookback"]),
            horizon=int(cfg["prediction_horizon"]),
            action_bins=int(cfg["action_bins"]),
            max_abs_return=float(cfg["max_abs_return"]),
        )
    return factory


def train(config_path: str, timesteps: int | None = None, days: int | None = None, device: str | None = None):
    cfg = load_config(config_path)
    if timesteps is not None:
        cfg["total_timesteps"] = timesteps
    if days is not None:
        cfg["days"] = days
    if device is not None:
        cfg["device"] = device
    np.random.seed(int(cfg["seed"]))
    symbols = [s.upper() for s in cfg["symbols"]]
    print(f"Downloading {cfg['days']} days in bounded REST pages; raw pages are not saved...")
    frames = [build_features(fetch_klines(s, cfg["interval"], int(cfg["days"]))) for s in symbols]
    train_frames, eval_frames = [], []
    for frame in frames:
        cut = int(len(frame) * 0.85)
        train_frames.append(frame.iloc[:cut].reset_index(drop=True))
        eval_frames.append(frame.iloc[cut:].reset_index(drop=True))
    mean, std = fit_normalizer(train_frames)
    train_vec = DummyVecEnv([
        make_env(f, mean, std, i, len(symbols), cfg) for i, f in enumerate(train_frames)
    ])
    eval_vec = DummyVecEnv([
        make_env(f, mean, std, i, len(symbols), cfg) for i, f in enumerate(eval_frames)
    ])
    out = Path(cfg["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = PPO(
        "MlpPolicy",
        train_vec,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=8,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 128]},
        verbose=1,
        seed=int(cfg["seed"]),
        device=cfg["device"],
    )
    callback = EvalCallback(
        eval_vec,
        best_model_save_path=str(out / "best"),
        log_path=str(out / "eval"),
        eval_freq=max(5000 // len(symbols), 1000),
        n_eval_episodes=3,
        deterministic=True,
    )
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=callback, progress_bar=False)
    model.save(out / "ppo_price_predictor")
    metadata = {
        "symbols": symbols,
        "interval": cfg["interval"],
        "lookback": int(cfg["lookback"]),
        "prediction_horizon": int(cfg["prediction_horizon"]),
        "feature_columns": FEATURE_COLUMNS,
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "action_bins": np.linspace(
            -float(cfg["max_abs_return"]), float(cfg["max_abs_return"]), int(cfg["action_bins"])
        ).tolist(),
        "training_rows": {s: len(f) for s, f in zip(symbols, train_frames)},
        "paper_only": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model and metadata under {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train the PPO crypto return predictor")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--days", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    train(args.config, args.timesteps, args.days, args.device)


if __name__ == "__main__":
    main()
