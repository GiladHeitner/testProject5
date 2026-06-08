#!/usr/bin/env bash
set -euo pipefail

# Qwen3-TTS modes (see assets/qwen_voice.json):
#   custom — preset speaker + style instruct (default, best for Omar Shorts)
#   design — voice from text description only (Qwen3-TTS-12Hz-1.7B-VoiceDesign)
#   clone  — copy a reference wav/m4a (old behavior)

QWEN_VOICE_MODE="${QWEN_VOICE_MODE:-custom}"
REF_AUDIO="${REF_AUDIO:-../assets/grove_3.m4a}"
REF_TEXT_FILE="${REF_TEXT_FILE:-transcript.txt}"
TEXT="${TEXT:-The gilded age of the late nineteenth century was one of the most}"
LANGUAGE="${LANGUAGE:-English}"
SPEAKER="${SPEAKER:-Ryan}"
VOICE_INSTRUCT="${VOICE_INSTRUCT:-Male 17-year-old Muslim Arab teen, energetic YouTube Shorts storyteller, casual rant delivery, engaging and hyped, clear articulation}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
OUTPUT="${OUTPUT:-custom.wav}"
SPEED="${SPEED:-1.12}"
PITCH="${PITCH:-0.0}"
USE_BATCH="${USE_BATCH:-false}"
SPEED_VALUES="${SPEED_VALUES:-0.85,1.0,1.15}"
PITCH_VALUES="${PITCH_VALUES:--2,0,2}"

cd "$(dirname "$0")"
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
elif [[ -f "../.venv/bin/activate" ]]; then
  source "../.venv/bin/activate"
fi

QWEN_MODEL_ID="$MODEL"
if [[ -n "${QWEN_MODEL_PATH:-}" && -d "$QWEN_MODEL_PATH" ]]; then
  MODEL="$QWEN_MODEL_PATH"
  echo "Using QWEN_MODEL_PATH: $MODEL"
elif [[ "${CI:-}" == "true" || -n "${GITHUB_ACTIONS:-}" ]]; then
  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  MODEL_CACHE="$HF_CACHE/models--$(echo "$QWEN_MODEL_ID" | tr '/' '--')/snapshots"
  if [[ -d "$MODEL_CACHE" ]]; then
    MODEL="$(find "$MODEL_CACHE" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
    echo "Using local Qwen snapshot: $MODEL"
  else
    MODEL="$QWEN_MODEL_ID"
  fi
fi

REF_TEXT=""
REF_FOR_CLONE=""
REF_TMP=""
if [[ "$QWEN_VOICE_MODE" == "clone" ]]; then
  REF_TEXT="$(cat "$REF_TEXT_FILE")"
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
fi

ARGS=(
  --mode "$QWEN_VOICE_MODE"
  --text "$TEXT"
  --language "$LANGUAGE"
  --model "$MODEL"
  --output "$OUTPUT"
  --speaker "$SPEAKER"
  --instruct "$VOICE_INSTRUCT"
)
if [[ "$QWEN_VOICE_MODE" == "clone" ]]; then
  ARGS+=(--ref-audio "$REF_FOR_CLONE" --ref-text "$REF_TEXT")
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ "$USE_BATCH" == "true" ]]; then
  python clone_voice.py \
    "${ARGS[@]}" \
    --speed-values "$SPEED_VALUES" \
    --pitch-values "$PITCH_VALUES"
else
  python clone_voice.py \
    "${ARGS[@]}" \
    --speed "$SPEED" \
    --pitch "$PITCH"
fi

echo "Done ($QWEN_VOICE_MODE mode, speaker=${SPEAKER})."
