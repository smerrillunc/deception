#!/usr/bin/env bash

###############################################
# Monte Carlo batch runner for BS game
# - Prompts for GPU selection
# - Runs MC on all seeds in results folder
# - Accepts extra args for mc_run_script.py
###############################################

# ---------------------------
# Activate conda environment
# ---------------------------
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

# Validate input (numeric)
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid GPU ID: $GPU"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# ---------------------------
# Extra arguments for Python
# ---------------------------
echo ""
read -p "Extra args for mc_run_script.py (or press Enter for none): " EXTRA_ARGS

# ---------------------------
# Paths
# ---------------------------
RESULT_PATH=/playpen-ssd/smerrill/deception/BS/results
SCRIPT=/playpen-ssd/smerrill/deception/BS/src/mc_run_script.py

# ---------------------------
# Run the Python MC script
# ---------------------------
printf "\n========== Running Monte Carlo on folder: %s ==========\n" "$RESULT_PATH"

# shellcheck disable=SC2086
python "$SCRIPT" \
    --result_path "$RESULT_PATH" \
    $EXTRA_ARGS

echo -e "\nAll Monte Carlo runs completed."
