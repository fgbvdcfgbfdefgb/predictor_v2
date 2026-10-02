# predictor_v2

A reproducible reinforcement-learning baseline for **one-minute BTC, ETH, and LTC price-return prediction**. It trains PPO to choose a discrete next-return forecast and rewards forecast accuracy with a Huber-error objective. A live Binance WebSocket process emits paper predictions after each closed 1-minute candle.

> **Research / paper mode only.** This repository never places orders and is not financial advice. Short-horizon crypto prices are noisy; backtest performance does not imply future profit.

## What it does

- Assets: `BTCUSDT`, `ETHUSDT`, `LTCUSDT`
- Input: 1-minute OHLCV candles and nine stationary technical features
- Agent: Stable-Baselines3 PPO, trained in three parallel asset environments
- Action: one of 11 predicted log-return bins
- Reward: `1 - 500 × Huber(predicted_return - realized_return)`
- Validation: chronological 85/15 split and periodic deterministic evaluation
- Live mode: Binance WebSocket, one prediction per closed candle, JSONL paper log
- Storage-aware: REST data is downloaded in bounded pages, retained in RAM, and raw data is never written to disk

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
predictor-train --config config/default.yaml --device cpu
predictor-live --config config/default.yaml
```

For a smoke run:

```bash
predictor-train --config config/default.yaml --days 2 --timesteps 10000 --device cpu
```

Artifacts are written to ignored `artifacts/`:

- `ppo_price_predictor.zip`
- `best/best_model.zip` when evaluation runs
- `metadata.json` with normalizer values and action bins
- `paper_predictions.jsonl` in live mode

## Configuration

Edit `config/default.yaml`. The production baseline uses 30 days and 250,000 steps. On a machine whose GPUs are occupied by another workload, pass `--device cpu`; the code does not terminate or modify unrelated processes.

## Deployment

```bash
./scripts/bootstrap.sh
./scripts/train_storage_safe.sh
./scripts/run_live.sh
```

The training wrapper forces CPU by default so a colocated GPU workload is not disturbed. Override with `PREDICTOR_DEVICE=cuda` only when GPU capacity is intentionally available.

## Limitations

RL is not inherently superior to supervised forecasting. This formulation is useful when forecast choices are discrete, but it should be compared against naive persistence, linear, and sequence-model baselines before any trading decision. Binance availability varies by jurisdiction. The live runner reconnects automatically but does not fill missed candles during a disconnect.

## Security

Never commit API keys, exchange credentials, passwords, model artifacts, or prediction logs. No exchange key is needed because only public market data is used.

## License

MIT
