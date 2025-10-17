#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate || true
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python run.py examples/mwe_a_2025-109.yaml > out1.txt
python run.py examples/mwe_a_2025-109.yaml > out2.txt
diff -q out1.txt out2.txt && echo "Determinism OK"
