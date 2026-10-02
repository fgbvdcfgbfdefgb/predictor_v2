#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
DEVICE="${PREDICTOR_DEVICE:-cpu}"
# Raw pages are never saved; artifacts are compact model/metadata files only.
exec predictor-train --config config/default.yaml --device "$DEVICE" "$@"
