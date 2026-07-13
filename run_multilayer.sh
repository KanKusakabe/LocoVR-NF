#!/usr/bin/env bash
# Reproduce the multi-layer NF experiments (Track B + Track H + comparison).
# Requires the LocoReal/LocoVR data (locovrnf.fetch) and the parent .venv.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
RUN="uv run --no-project python -m"

echo "== Track B (product / particle filter) =="
$RUN locovrnf.filter --holdout 3            # B0/B1/B2 -> results/b_filter.json + .png

echo "== Track H (coarse->fine hierarchy) =="
$RUN locovrnf.coarse --holdout 3 --loo      # H0/H1/H2/H3 -> results/h_coarse.json + .png

echo "== X1 comparison =="
$RUN locovrnf.compare                        # X1 table -> results/x_compare.json + .png

echo "done. See README.md and index.html."
