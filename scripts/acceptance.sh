#!/usr/bin/env bash
# Reproducible Section 16 acceptance checks. Run from the repository root.
set -euo pipefail

python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
# Normal clean-environment acceptance runs execute every test. Set
# SKIP_API_TESTS=1 only on a constrained host that cannot create TestClient sockets.
if [[ "${SKIP_API_TESTS:-0}" == "1" ]]; then
  python -m pytest -q --ignore=tests/test_api.py
else
  python -m pytest -q
fi
python -m pytest tests/test_overfit.py -q

# This script loads minigpt_final.pt in a fresh Python process, samples three
# fixed prompts, and proves a future-token perturbation leaves earlier logits unchanged.
python -m experiments.m5_acceptance_checks --output outputs/logs/acceptance_m5_checks.txt

for required in "python -m pytest" "python -m src.train" "python -m src.generate"; do
  if ! grep -Fq "$required" README.md; then
    echo "README is missing documented command: $required" >&2
    exit 1
  fi
done
for required_path in "src/train.py" "src/generate.py" "tests/test_overfit.py"; do
  if [[ ! -f "$required_path" ]]; then
    echo "README command target is missing from the repository: $required_path" >&2
    exit 1
  fi
done

# Local secrets are forbidden. Exclude generated outputs and this scanner itself
# so the regex literals are not reported as a false positive.
if find . -type f \( -name '.env' -o -name '*.env' \) ! -name '.env.example' -print -quit | grep -q .; then
  echo "Found a local .env file. Remove it before publishing." >&2
  exit 1
fi
if rg -n \
  --glob '!\.git/**' --glob '!outputs/**' --glob '!scripts/acceptance.sh' \
  -e 'AKIA[0-9A-Z]{16}' -e 'sk-[A-Za-z0-9_-]{20,}' \
  -e 'gh[pousr]_[A-Za-z0-9]{30,}' -e '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----' .; then
  echo "Possible secret found. Remove it before publishing." >&2
  exit 1
fi

echo "Acceptance checks passed."
