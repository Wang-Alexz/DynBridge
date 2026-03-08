SNAPSHOT_DIR="/your_cppath/snapshot"
CHECKED_LIST="${SNAPSHOT_DIR}/checked_lists.txt"

touch "$CHECKED_LIST"

while true; do
    echo "Checking for new checkpoints at $(date)..."
    
    for pt_path in $(ls ${SNAPSHOT_DIR}/*.pt 2>/dev/null | sort -V); do
        filename=$(basename -- "$pt_path")
        step="${filename%.*}"

        if grep -qx "$filename" "$CHECKED_LIST"; then
            continue
        fi

        if [ "$step" -lt 5000 ]; then
            echo "Skip $filename (step=$step < 5000)"
            echo "$filename" >> "$CHECKED_LIST"
            continue
        fi

        echo "Evaluating checkpoint: $filename"

        output_dir="${SNAPSHOT_DIR}/${step}"
        mkdir -p "$output_dir"

        MUJOCO_GL=glx xvfb-run --auto-servernum \
        python eval_ray.py suite/task=libero_objecttest suite.track_ts=15 alpha=0.5 hydra.run.dir="${output_dir}" bc_weight="${pt_path}"

        echo "$filename" >> "$CHECKED_LIST"
    done

    echo "Sleeping 10 minutes..."
    sleep 600   
done
