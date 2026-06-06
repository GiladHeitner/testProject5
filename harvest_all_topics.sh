#!/usr/bin/env bash
# Run many harvest batches slowly (avoids PullPush rate limits).
# Usage: ./harvest_all_topics.sh [first_batch] [last_batch]
# Example: ./harvest_all_topics.sh 0 11   → ~12 batches, ~45s pause each
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

START="${1:-0}"
END="${2:-11}"
BATCH_PAUSE="${BATCH_PAUSE:-45}"

chmod +x "$ROOT_DIR/harvest_topics.sh"

for b in $(seq "$START" "$END"); do
  echo ""
  echo "========== Harvest batch $b / $END =========="
  if ! "$ROOT_DIR/harvest_topics.sh" "$b"; then
    echo "[warn] Batch $b failed (rate limit or no new topics). Continuing…" >&2
  fi
  if [[ "$b" -lt "$END" ]]; then
    echo "Pausing ${BATCH_PAUSE}s before next batch…"
    sleep "$BATCH_PAUSE"
  fi
done

echo ""
python3 -c "from pathlib import Path; from shorts_bot_lib.reddit_topics import load_topic_entries; print('topics.txt:', len(load_topic_entries(Path('topics.txt'))), 'entries')"
