#!/bin/bash

MODEL_NAME="t5-large"
START_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/pc"
VMSTAT_LOG="${LOG_DIR}/${MODEL_NAME}_${START_TIME}_vmstat.log"
NVSMI_LOG="${LOG_DIR}/${MODEL_NAME}_${START_TIME}_nvsmi.log"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Start vmstat logging
( while true; do date +"%Y-%m-%d %H:%M:%S"; vmstat 1 1; done ) | awk 'NR==1{header=$0; next} NR%3==2{printf "%s ", prev} {print} {prev=$0}' > "$VMSTAT_LOG" &
VMSTAT_PID=$!

# Start nvidia-smi logging
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > "$NVSMI_LOG" &
NVSMI_PID=$!

# Run your model
python t5-latency.py --model_name "$MODEL_NAME"

# Stop logging
kill $VMSTAT_PID
kill $NVSMI_PID

echo "vmstat logged to $VMSTAT_LOG"
echo "nvidia-smi logged to $NVSMI_LOG"