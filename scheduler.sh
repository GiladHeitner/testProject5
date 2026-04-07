#!/bin/bash

# Paths
PROJECT_DIR="/Users/giladheitner/Documents/VideoBots"
RUN_SCRIPT="$PROJECT_DIR/run.sh"
STATE_FILE="$PROJECT_DIR/scheduler_state.txt"
LOG_FILE="$PROJECT_DIR/scheduler.log"

# Get current time info
TODAY=$(date +"%Y-%m-%d")
NOW_H=$(date +"%H")
NOW_M=$(date +"%M")
# Convert to total minutes for easy math (10# prevents octal errors with 08/09)
NOW_MINUTES=$(( 10#$NOW_H * 60 + 10#$NOW_M ))

# Read state if it exists
if [ -f "$STATE_FILE" ]; then
    read -r STATE_DATE TARGET_H TARGET_M STATUS < "$STATE_FILE"
else
    STATE_DATE=""
fi

# If it's a new day, pick a new random time
if [ "$STATE_DATE" != "$TODAY" ]; then
    # Random hour between 10 and 20 (10 AM to 8 PM)
    TARGET_H=$(( RANDOM % 11 + 10 ))
    # Random minute between 0 and 59
    TARGET_M=$(( RANDOM % 60 ))
    STATUS="pending"

    # Pad with zeros for logging
    TARGET_H_PAD=$(printf "%02d" $TARGET_H)
    TARGET_M_PAD=$(printf "%02d" $TARGET_M)

    echo "$TODAY $TARGET_H $TARGET_M $STATUS" > "$STATE_FILE"
    echo "$(date): New day. Target time set to $TARGET_H_PAD:$TARGET_M_PAD" >> "$LOG_FILE"
fi

# If pending, check if it's time to run
if [ "$STATUS" == "pending" ]; then
    TARGET_MINUTES=$(( 10#$TARGET_H * 60 + 10#$TARGET_M ))

    if [ "$NOW_MINUTES" -ge "$TARGET_MINUTES" ]; then
        echo "$(date): Target time reached. Running video bot..." >> "$LOG_FILE"

        # Navigate to the directory so relative paths in run.sh work
        cd "$PROJECT_DIR" || exit
        
        # Run the script and log the output
        bash "$RUN_SCRIPT" >> "$LOG_FILE" 2>&1

        # Mark as done for today
        echo "$TODAY $TARGET_H $TARGET_M done" > "$STATE_FILE"
        echo "$(date): Run complete. Marked as done." >> "$LOG_FILE"
    fi
fi
