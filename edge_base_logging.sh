#!/bin/bash

MODEL_LIST=("t5-small" "t5-base" "t5-large")

for MODEL_NAME in "${MODEL_LIST[@]}";
do
	
	START_TIME=$(date +"%Y%m%d_%H%M%S")
	LOG_DIR="logs/edge/${MODEL_NAME}"
	LOG_FILE_TGS="${LOG_DIR}/${MODEL_NAME}_${START_TIME}_tegrastats.log"
	LOG_FILE_TMN="${LOG_DIR}/${MODEL_NAME}_${START_TIME}_terminal.log"

	# Ensure the log directory exists
	mkdir -p "$LOG_DIR"
	# Ensure the log file is created in the log directory
	if [ ! -d "$LOG_DIR" ]; then
	    mkdir -p "$LOG_DIR"
	fi

	tegrastats --interval 1000 > "$LOG_FILE_TGS" &
	TEGRASTATS_PID=$!

	python t5-latency.py --model_name "$MODEL_NAME" --device "cuda" --runs 500 | tee "$LOG_FILE_TMN"
	TEGRA_PID=$!
	kill $TEGRASTATS_PID

	echo "Tegra stats logged to $LOG_FILE"
done
