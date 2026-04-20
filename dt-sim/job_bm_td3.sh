#!/bin/bash
#SBATCH --job-name=duckie_rl_td3
#SBATCH --output=output/duckie_%j.out
#SBATCH -e output/duckie_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

# using my wand account
export WANDB_API_KEY=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
export WANDB_DIR=$PWD/wandb


echo "Using Python from: $(which python)"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYGLET_DEBUG_GL=False
export PYGLET_HEADLESS=True

if [ ! -f $CONDA_PREFIX/lib/libtiff.so.5 ]; then
    ln -s $CONDA_PREFIX/lib/libtiff.so.6 $CONDA_PREFIX/lib/libtiff.so.5
fi

# --- 4. Launch Training ---
python rl_bm/td3_continuous_action.py \
    --seed 123 \
    --env-id oval_loop \
    --total-timesteps 1000000 \
    --buffer-size 5000 \
    --learning-starts 5000 \
    --run-notes "First test run of td3"
