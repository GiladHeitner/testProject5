#!/usr/bin/env bash
# Scrape Reddit on your Mac → topics.txt → commit & push for GitHub Actions.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

LIMIT="${LIMIT:-25}"
python3 -m shorts_bot_lib.reddit_topics sync-topics --limit "$LIMIT" --out topics.txt

echo ""
echo "Next: review topics.txt, then push to GitHub:"
echo "  git add topics.txt"
echo "  git commit -m 'chore: sync topics from Reddit'"
echo "  git push origin main"
