#!/usr/bin/env bash
# Full run on the real Bossard files that sit one folder above this repo.
# Usage:  bash run_bossard.sh            (extra flags are passed through, e.g. --memory-limit 4GB)
cd "$(dirname "$0")"
PYTHONPATH=src python3 -m smartbin run \
  --measurements ../measurements.csv \
  --metadata ../scale_metadata.csv \
  --preprocessing ../scale_preprocessing.csv \
  --reports reports/bossard "$@"
