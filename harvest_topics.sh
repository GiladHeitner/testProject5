#!/usr/bin/env bash
# Add real Reddit story topics per batch (merge into topics.txt).
# Usage: ./harvest_topics.sh [batch_number]
# Run batches 0,1,2,3… or use ./harvest_all_topics.sh for slow bulk harvest.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

BATCH="${1:-0}"
LIMIT="${LIMIT:-15}"

# Slow defaults — PullPush rate-limits fast scrapers.
export REDDIT_FETCH_DELAY="${REDDIT_FETCH_DELAY:-6}"
export REDDIT_HTTP_TIMEOUT="${REDDIT_HTTP_TIMEOUT:-60}"
export REDDIT_HARVEST_BATCH_SIZE="${REDDIT_HARVEST_BATCH_SIZE:-2}"
PER_SOURCE="${PER_SOURCE:-25}"

if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

python3 -m shorts_bot_lib.reddit_topics harvest-topics \
  --limit "$LIMIT" \
  --merge \
  --batch "$BATCH" \
  --source pullpush \
  --per-source "$PER_SOURCE"

echo ""
echo "Done batch $BATCH. Next: ./harvest_topics.sh $((BATCH + 1))"
