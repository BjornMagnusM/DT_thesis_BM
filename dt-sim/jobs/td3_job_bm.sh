#!/bin/bash
#SBATCH --job-name=Mid3Ter3_100
#SBATCH --output=output/duckie_%j.out
#SBATCH --error=output/duckie_%j.err
#SBATCH --time=76:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate duckie-rl

export WANDB_API_KEY=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
export WANDB_DIR=$PWD/wandb

echo "Using Python from: $(which python)"

# IMPORTANT: only project root
export PYTHONPATH="$PWD"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYGLET_DEBUG_GL=False
export PYGLET_HEADLESS=True

if [ ! -f $CONDA_PREFIX/lib/libtiff.so.5 ]; then
    ln -s $CONDA_PREFIX/lib/libtiff.so.6 $CONDA_PREFIX/lib/libtiff.so.5
fi

python3  rl_bm/td3_continuous_action.py \
    --seed 2 \
    --env-id oval_loop \
    --total-timesteps 1000000 \
    --buffer-size 10000 \
    --learning-starts 10000 \
    --time_optimal_reward \
    --lap_termination \
    --run-notes "Mid3 Ter3_100"