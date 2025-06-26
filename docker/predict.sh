#!/usr/bin/env bash
# predict.sh

set -euo pipefail

# FLARE mounts the test scans at /input and expects your NIfTI outputs in /output
python inference.py \
  --inputs_dir "/input" \
  --output_dir "/output" \
  --device "cpu"
