from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ReturnPredictionEnv(gym.Env):
    """RL environment where actions are next-return bins.

    Reward is the negative Huber error between the selected return bin and the
    realized future log return. This makes the RL objective explicitly a price
    forecast objective rather than an order-execution strategy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        closes: np.ndarray,
        asset_index: int,
        asset_count: int,
        lookback: int = 60,
        horizon: int = 1,
        action_bins: int = 11,
        max_abs_return: float = 0.012,
        episode_steps: int = 2048,
    ) -> None:
        super().__init__()
        if len(features) <= lookback + horizon + 1:
            raise ValueError("Not enough rows for lookback and prediction horizon")
        self.features = features.astype(np.float32)
        self.closes = closes.astype(np.float64)
        self.asset_index = asset_index
        self.asset_count = asset_count
        self.lookback = lookback
        self.horizon = horizon
        self.return_bins = np.linspace(-max_abs_return, max_abs_return, action_bins).astype(np.float32)
        self.episode_steps = min(episode_steps, len(features) - lookback - horizon)
        obs_size = lookback * features.shape[1] + asset_count
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(action_bins)
        self._i = lookback
        self._end = len(features) - horizon

    def _obs(self) -> np.ndarray:
        window = self.features[self._i - self.lookback : self._i].reshape(-1)
        asset = np.zeros(self.asset_count, dtype=np.float32)
        asset[self.asset_index] = 1.0
        return np.concatenate([window, asset]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        latest_start = max(self.lookback, len(self.features) - self.horizon - self.episode_steps - 1)
        self._i = int(self.np_random.integers(self.lookback, latest_start + 1))
        self._end = min(self._i + self.episode_steps, len(self.features) - self.horizon)
        return self._obs(), {}

    def step(self, action: int):
        predicted = float(self.return_bins[int(action)])
        actual = float(np.log(self.closes[self._i + self.horizon] / self.closes[self._i]))
        error = predicted - actual
        delta = 0.002
        huber = 0.5 * error * error / delta if abs(error) <= delta else abs(error) - 0.5 * delta
        reward = 1.0 - 500.0 * huber
        self._i += 1
        terminated = self._i >= self._end
        info = {
            "predicted_return": predicted,
            "actual_return": actual,
            "absolute_error": abs(error),
        }
        return self._obs(), float(reward), terminated, False, info
