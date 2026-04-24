#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    echo "Created .env from .env.example. Add your API keys, then run again."
  else
    echo "Missing .env."
  fi
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source ".venv/bin/activate"

python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install -r requirements.txt >/dev/null 2>&1

set -a
source .env
set +a

exec python ui.py "$@"
