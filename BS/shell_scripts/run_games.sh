#!/usr/bin/env bash

###############################################
# Interactive BS game runner
# - Prompts for GPU selection
# - Runs seeds sequentially
# - Accepts extra args for the Python script
###############################################

echo "Activating conda environment: opensloth_env"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh   
conda activate deception

# ---------------------------
# Ask user for GPU selection
# ---------------------------
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
read -p "Enter the GPU ID you want to use (e.g., 0): " GPU

# Validate input (numeric)
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid GPU ID: $GPU"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""


# ---------------------------
# Get seed range
# ---------------------------
read -p "Enter start seed: " SEED_START
read -p "Enter end seed:   " SEED_END

if ! [[ "$SEED_START" =~ ^[0-9]+$ && "$SEED_END" =~ ^[0-9]+$ ]]; then
    echo "❌ Seeds must be integers."
    exit 1
fi


# ---------------------------
# Extra arguments for Python
# ---------------------------
echo ""
read -p "Extra args for run_game.py (or press Enter for none): " EXTRA_ARGS

logdir=/playpen-ssd/smerrill/deception/BS/results
script=/playpen-ssd/smerrill/deception/BS/src/single_run_script.py
# ---------------------------
# Run seeds sequentially
# ---------------------------
for (( seed=$SEED_START; seed <= $SEED_END; seed++ )); do
    printf "\n========== Running seed %04d ==========\n" "$seed"

    # shellcheck disable=SC2086
    python "$script" \
        --seed "$seed" \
        --log-root "$logdir" \
        $EXTRA_ARGS

    echo "✓ Finished seed $seed"
done

echo -e "\nAll runs completed."