#!/bin/bash
# Copies episodes from each task folder into a new folder with episode numbers offset by 50.
# e.g. episode0.pkl -> episode50.pkl, episode99.pkl -> episode149.pkl

BASE_DIR="/home/weirdlab/Documents/summers/IsaacLab/source/recorded_runs/KITCHEN_TASK/UNPAIRED_DATA/FRANKA"
OFFSET=50

for task_dir in "$BASE_DIR"/*/; do
    task_name=$(basename "$task_dir")
    new_dir="$BASE_DIR/new${task_name}"
    mkdir -p "$new_dir"

    for f in "$task_dir"episode*; do
        filename=$(basename "$f")
        # Extract number and extension from e.g. episode42.pkl
        num=$(echo "$filename" | sed 's/episode\([0-9]*\)\..*/\1/')
        ext="${filename##*.}"
        new_num=$((num + OFFSET))
        cp "$f" "$new_dir/episode${new_num}.${ext}"
    done

    echo "Done: $task_name -> new${task_name} ($(ls "$new_dir" | wc -l) files)"
done
