#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Edit these values each run
# -----------------------------
REF_AUDIO="${REF_AUDIO:-grove_3.wav}"
REF_TEXT_FILE="${REF_TEXT_FILE:-transcript.txt}"
TEXT="${TEXT:-The gilded age of the late nineteenth century was one of the most}"
LANGUAGE="${LANGUAGE:-English}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
OUTPUT="${OUTPUT:-custom.wav}"

# Single tweak mode (Omar reference is already teen male — keep pitch neutral):
SPEED="${SPEED:-1.20}"   # 1.0 = normal, <1 slower, >1 faster
PITCH="${PITCH:-0.0}"    # semitones, e.g. -2, 0, +2

# Batch test mode (set USE_BATCH=true to enable):
USE_BATCH="${USE_BATCH:-false}"
SPEED_VALUES="${SPEED_VALUES:-0.85,1.0,1.15}"
PITCH_VALUES="${PITCH_VALUES:--2,0,2}"

# -----------------------------
# Script logic
# -----------------------------
cd "$(dirname "$0")"
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
elif [[ -f "../.venv/bin/activate" ]]; then
  source "../.venv/bin/activate"
fi

REF_TEXT="$(cat "$REF_TEXT_FILE")"

if [[ "$USE_BATCH" == "true" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  python clone_voice.py \
    --ref-audio "$REF_AUDIO" \
    --ref-text "$REF_TEXT" \
    --text "$TEXT" \
    --language "$LANGUAGE" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --speed-values "$SPEED_VALUES" \
    --pitch-values "$PITCH_VALUES"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  python clone_voice.py \
    --ref-audio "$REF_AUDIO" \
    --ref-text "$REF_TEXT" \
    --text "$TEXT" \
    --language "$LANGUAGE" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --speed "$SPEED" \
    --pitch "$PITCH"
fi

echo "Done."
