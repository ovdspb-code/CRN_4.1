#!/usr/bin/env bash
set -euo pipefail

# Smoke test for CRN_Sim (R12-CROWN)
#
# Runs:
#   1) import checks
#   2) R12-CROWN/run_all.py
#   3) R12-CROWN/validate_checkpoints.py

python - <<'PY'
import sys
print('Python:', sys.version)
mods=['numpy','scipy','pandas','matplotlib','yaml']
for m in mods:
    __import__(m)
print('Imports: OK')
PY

python R12-CROWN/run_all.py
python R12-CROWN/validate_checkpoints.py
