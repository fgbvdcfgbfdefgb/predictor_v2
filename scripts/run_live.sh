#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p artifacts
exec predictor-live --config config/default.yaml
