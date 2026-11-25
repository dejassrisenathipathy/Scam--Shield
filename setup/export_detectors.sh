#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m ml.prepare_sms
python -m ml.train_sms
python -m ml.export_sms
