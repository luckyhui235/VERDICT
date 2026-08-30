#!/usr/bin/env bash

if [ -f "./.env" ]; then
    export $(grep -v '^#' "./.env" | xargs)
fi

# Logs
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%F_%H%M%S)
LOG_FILE="$LOG_DIR/run_mas_${TIMESTAMP}.log"


nohup python legal-task/run.py \
    --task cail2018 \
    --reasoning io \
    --mas_memory case-memory \
    --max_trials 6 \
    --mode test \
    --mas_type macnet \
    --model deepseek-chat \
    --successful_topk 2 \
    --threshold 0.3 \
    "$@" > "$LOG_FILE" 2>&1 &

echo "Started run_mas (PID: $!) | Logs: $LOG_FILE"

# To run CAIL2018, change --task to cail2018
# python3 legal-task/run.py \
#   --task cail2018 \
#   --reasoning io \
#   --mas_memory case-memory \
#   --max_trials 12 \
#   --mas_type macnet \
#   --model Qwen/Qwen2.5-14B-Instruct
