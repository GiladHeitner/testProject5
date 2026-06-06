#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Edit these values each run
# -----------------------------
REF_AUDIO="${REF_AUDIO:-../assets/grove_3.m4a}"
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

# Qwen expects wav; convert m4a/mp3/etc. on the fly.
REF_FOR_CLONE="$REF_AUDIO"
REF_TMP=""
if [[ ! -f "$REF_AUDIO" ]]; then
  echo "Reference audio not found: $REF_AUDIO" >&2
  exit 1
fi
case "$REF_AUDIO" in
  *.wav|*.WAV) REF_FOR_CLONE="$REF_AUDIO" ;;
  *)
    REF_TMP="$(mktemp "${TMPDIR:-/tmp}/omar_ref.XXXXXX.wav")"
    ffmpeg -y -loglevel error -i "$REF_AUDIO" -ar 24000 -ac 1 "$REF_TMP"
    REF_FOR_CLONE="$REF_TMP"
    ;;
esac
cleanup_ref() {
  if [[ -n "$REF_TMP" && -f "$REF_TMP" ]]; then
    rm -f "$REF_TMP"
  fi
}
trap cleanup_ref EXIT

QWEN_MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_CACHE="$HF_CACHE/models--$(echo "$QWEN_MODEL" | tr '/' '--')"
if [[ "${CI:-}" == "true" || -n "${GITHUB_ACTIONS:-}" ]]; then
  if [[ -d "$MODEL_CACHE" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "Using cached Qwen model offline: $QWEN_MODEL"
  fi
fi

if [[ "$USE_BATCH" == "true" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  python clone_voice.py \
    --ref-audio "$REF_FOR_CLONE" \
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
    --ref-audio "$REF_FOR_CLONE" \
    --ref-text "$REF_TEXT" \
    --text "$TEXT" \
    --language "$LANGUAGE" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --speed "$SPEED" \
    --pitch "$PITCH"
fi

echo "Done."
