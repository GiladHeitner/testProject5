#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ first."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install ffmpeg first."
  exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe not found. Install ffmpeg tools first."
  exit 1
fi

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    echo "Created .env from .env.example. Add your API keys, then run again."
  else
    echo "Missing .env and .env.example."
  fi
  exit 1
fi

mkdir -p assets/gameplay assets/popups assets/story_images output

shopt -s nullglob
gameplay_files=(assets/gameplay/*.mp4 assets/gameplay/*.mov assets/gameplay/*.mkv assets/gameplay/*.webm)
shopt -u nullglob
if [[ ${#gameplay_files[@]} -eq 0 ]]; then
  echo "No gameplay video found in assets/gameplay."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source ".venv/bin/activate"
python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install -r requirements.txt >/dev/null 2>&1

WORDS="${WORDS:-100}"

upload_args=(--upload --no-description)
forward_args=()
is_quick_test=0
topic_arg=()

for arg in "$@"; do
  if [[ "$arg" == "--no-upload" ]]; then
    upload_args=()
  elif [[ "$arg" == "--quick-test" ]]; then
    is_quick_test=1
    upload_args=()
    forward_args+=("$arg")
  else
    forward_args+=("$arg")
  fi
done

has_topic=0
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--topic" ]]; then
    has_topic=1
    break
  fi
done

if [[ $is_quick_test -eq 0 && $has_topic -eq 0 && -t 0 ]]; then
  echo
  read -r -p "What's the topic? " entered_topic
  if [[ -n "${entered_topic// }" ]]; then
    topic_arg=(--topic "$entered_topic")
  fi
fi

cmd=(
  python shorts_bot.py
  --words "$WORDS"
  --gameplay-top-crop 96
  --bgm-volume 0.25
)

if [[ ${#topic_arg[@]} -gt 0 ]]; then
  cmd+=("${topic_arg[@]}")
fi

if [[ ${#upload_args[@]} -gt 0 ]]; then
  cmd+=("${upload_args[@]}")
fi

if [[ ${#forward_args[@]} -gt 0 ]]; then
  cmd+=("${forward_args[@]}")
fi

"${cmd[@]}"
