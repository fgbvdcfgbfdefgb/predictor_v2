from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "vol_10", "vol_30", "rsi_14",
    "volume_z_30", "range_frac", "body_frac",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stationary features and retain close for target construction."""
    df = frame.copy()
    close = df["close"].clip(lower=1e-12)
    log_close = np.log(close)
    ret = log_close.diff()
    df["ret_1"] = ret
    df["ret_3"] = log_close.diff(3) / 3.0
    df["ret_5"] = log_close.diff(5) / 5.0
    df["vol_10"] = ret.rolling(10).std()
    df["vol_30"] = ret.rolling(30).std()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = ((100 - 100 / (1 + rs)) - 50) / 50
    vol_mean = df["volume"].rolling(30).mean()
    vol_std = df["volume"].rolling(30).std().replace(0, np.nan)
    df["volume_z_30"] = (df["volume"] - vol_mean) / vol_std
    df["range_frac"] = (df["high"] - df["low"]) / close
    df["body_frac"] = (df["close"] - df["open"]) / close
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df[["open_time", "close", *FEATURE_COLUMNS]]


def fit_normalizer(frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([f[FEATURE_COLUMNS].to_numpy(np.float32) for f in frames])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalized_values(frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = frame[FEATURE_COLUMNS].to_numpy(np.float32)
    return np.clip((values - mean) / std, -10.0, 10.0).astype(np.float32)
