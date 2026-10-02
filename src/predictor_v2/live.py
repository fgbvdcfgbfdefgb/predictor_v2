from __future__ import annotations

import argparse
import asyncio
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import websockets

from .config import load_config
from .data import fetch_klines
from .features import build_features, normalized_values

BINANCE_WS = "wss://stream.binance.com:9443/ws"


class LivePredictor:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        model_dir = Path(self.cfg["model_dir"])
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        model_path = model_dir / "ppo_price_predictor.zip"
        best_path = model_dir / "best" / "best_model.zip"
        self.model = PPO.load(best_path if best_path.exists() else model_path, device="cpu")
        self.mean = np.array(self.meta["normalizer_mean"], dtype=np.float32)
        self.std = np.array(self.meta["normalizer_std"], dtype=np.float32)
        self.bins = np.array(self.meta["action_bins"], dtype=np.float32)
        self.symbols = self.meta["symbols"]
        self.rows: dict[str, deque] = {}
        self.log_path = Path(self.cfg["live"]["paper_log"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def warmup(self):
        for symbol in self.symbols:
            frame = fetch_klines(symbol, self.meta["interval"], days=1)
            self.rows[symbol] = deque(frame.tail(max(180, self.meta["lookback"] + 40)).to_dict("records"), maxlen=500)

    def predict(self, symbol: str) -> dict:
        frame = pd.DataFrame(self.rows[symbol])
        featured = build_features(frame)
        lookback = int(self.meta["lookback"])
        if len(featured) < lookback:
            raise RuntimeError(f"Warmup incomplete for {symbol}")
        values = normalized_values(featured, self.mean, self.std)[-lookback:].reshape(-1)
        one_hot = np.zeros(len(self.symbols), dtype=np.float32)
        one_hot[self.symbols.index(symbol)] = 1.0
        obs = np.concatenate([values, one_hot]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        predicted_return = float(self.bins[int(action)])
        current = float(featured.iloc[-1]["close"])
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "current_price": current,
            "predicted_return_1m": predicted_return,
            "predicted_price_1m": current * float(np.exp(predicted_return)),
            "mode": "paper_prediction_only",
        }

    def emit(self, result: dict):
        line = json.dumps(result, separators=(",", ":"))
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    async def stream_symbol(self, symbol: str):
        stream = f"{BINANCE_WS}/{symbol.lower()}@kline_{self.meta['interval']}"
        delay = int(self.cfg["live"]["reconnect_seconds"])
        while True:
            try:
                async with websockets.connect(stream, ping_interval=20, ping_timeout=20) as ws:
                    async for raw in ws:
                        event = json.loads(raw)
                        candle = event["k"]
                        if not candle["x"]:
                            continue
                        self.rows[symbol].append({
                            "open_time": pd.to_datetime(candle["t"], unit="ms", utc=True),
                            "open": float(candle["o"]), "high": float(candle["h"]),
                            "low": float(candle["l"]), "close": float(candle["c"]),
                            "volume": float(candle["v"]),
                        })
                        self.emit(self.predict(symbol))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"{symbol} stream error: {exc}; reconnecting in {delay}s", flush=True)
                await asyncio.sleep(delay)

    async def run(self):
        await asyncio.to_thread(self.warmup)
        await asyncio.gather(*(self.stream_symbol(s) for s in self.symbols))


def main():
    parser = argparse.ArgumentParser(description="Run live one-minute paper predictions")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    asyncio.run(LivePredictor(args.config).run())


if __name__ == "__main__":
    main()
