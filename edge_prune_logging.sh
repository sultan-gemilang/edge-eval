#!/bin/bash

MODEL_LIST=(
    "./nash_out/t5-small/SAMSUM/NASH/SAMSUM_nash_unif_0.5_3/int/FT/best"
    "./nash_out/t5-small/SAMSUM/NASH/SAMSUM_nash_unif_0.3_3/int/FT/best"
    "./nash_out/t5-base/SAMSUM/NASH/SAMSUM_nash_unif_0.3_3/int/FT/best"
    "./nash_out/t5-base/SAMSUM/NASH/SAMSUM_nash_unif_0.5_3/int/FT/best"
)

for MODEL_PATH in "${MODEL_LIST[@]}";
do
	if [[ $MODEL_PATH == *"t5-base"* ]]; then
		MODEL_TYPE="t5-base"
		MODEL_NAME=$(basename $(dirname $(dirname $(dirname "$MODEL_PATH"))))
	elif [[ $MODEL_PATH == *"t5-small"* ]]; then
		MODEL_TYPE="t5-small"
		MODEL_NAME=$(basename $(dirname $(dirname $(dirname "$MODEL_PATH"))))
	else
		echo "Unknown model type for $MODEL_PATH"
		exit 1
	fi

	START_TIME=$(date +"%Y%m%d_%H%M%S")
	LOG_DIR="logs/edge/${MODEL_TYPE}"
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

	python t5-samsum.py --model_name "$MODEL_NAME" --device "cuda" --runs 500 | tee "$LOG_FILE_TMN"
	TEGRA_PID=$!
	kill $TEGRASTATS_PID

	echo "Tegra stats logged to $LOG_FILE"
done
