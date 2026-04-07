#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install -r requirements.txt >/dev/null 2>&1

# Easy testing knobs (edit these values):
# Speed and pitch are separate controls.
VOICES="${VOICES:-alloy}"
SPEED="${SPEED:-1.8}"
PITCH_FACTOR="${PITCH_FACTOR:-1.4}"
TARGET_SECONDS="${TARGET_SECONDS:-0}"
OUT_DIR="${OUT_DIR:-output/voice_tests}"

echo "Voice test settings: VOICES=$VOICES SPEED=$SPEED PITCH_FACTOR=$PITCH_FACTOR TARGET_SECONDS=$TARGET_SECONDS"

python voice_test.py \
  --voices "$VOICES" \
  --speech-speed "$SPEED" \
  --pitch-factor "$PITCH_FACTOR" \
  --target-seconds "$TARGET_SECONDS" \
  --out-dir "$OUT_DIR" \
  "$@"
