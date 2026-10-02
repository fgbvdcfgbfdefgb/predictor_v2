#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts
cp models/smoke_2026-10-02/ppo_price_predictor.zip artifacts/
cp models/smoke_2026-10-02/metadata.json artifacts/
echo "Installed smoke checkpoint under artifacts/."
