# Smoke checkpoint model card

- Trained: 2026-10-02 UTC
- Assets: BTCUSDT, ETHUSDT, LTCUSDT
- Data: most recent 2 days available from Binance at training time
- Training: PPO on CPU, requested 10,000 steps (10,752 collected because of rollout size)
- Chronological split: 85% training / 15% evaluation
- Last deterministic evaluation mean reward: -531.59 over three episodes
- Purpose: deployment and integration smoke test only

This checkpoint is **not production-quality** and its negative evaluation reward is not evidence of predictive value. It is included so the live paper pipeline can be reproduced without retraining. Use `scripts/use_smoke_model.sh` to copy it into the ignored runtime artifact directory.
