#!/usr/bin/env bash
# Scrape full Reddit posts on your Mac → topics.txt → commit & push for GitHub Actions.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

LIMIT="${LIMIT:-25}"
EXTRA=()
if [[ "${REPLACE:-0}" == "1" ]]; then
  EXTRA+=(--replace)
fi
python3 -m shorts_bot_lib.reddit_topics sync-topics --limit "$LIMIT" --out topics.txt "${EXTRA[@]}"

echo ""
echo "Next: review topics.txt, then push to GitHub:"
echo "  git add topics.txt"
echo "  git commit -m 'chore: sync topics from Reddit'"
echo "  git push origin main"
