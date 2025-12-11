#!/usr/bin/env bash

############################################################
# Deception label extractor runner
# - Prompts for GPU selection
# - Runs seeds sequentially
# - Accepts extra args for the Python script
# - Uses deception_labels.py
############################################################

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

# ---------------------------
# Ask user for GPU selection
# ---------------------------
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
read -p "Enter the GPU ID you want to use (e.g., 0): " GPU

# Validate input
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
read -p "Extra args for deception_labels.py (or press Enter for none): " EXTRA_ARGS

# ---------------------------
# Paths
# ---------------------------
script=/playpen-ssd/smerrill/deception/BS/src/deception_labels.py
results_root=/playpen-ssd/smerrill/deception/BS/results

# ---------------------------
# Run seeds sequentially
# ---------------------------
for (( seed=$SEED_START; seed <= $SEED_END; seed++ )); do
    printf "\n========== Running deception extraction for seed %04d ==========\n" "$seed"

    seed_path="$results_root/game_seed_${seed}"

    if [ ! -d "$seed_path" ]; then
        echo "⚠️ Seed path does not exist: $seed_path — skipping"
        continue
    fi

    # shellcheck disable=SC2086
    python "$script" \
        --seed_path "$seed_path" \
        --model "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
        $EXTRA_ARGS

    echo "✓ Finished seed $seed"
done

echo -e "\nAll deception labeling completed."