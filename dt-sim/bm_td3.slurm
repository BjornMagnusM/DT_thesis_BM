#!/bin/bash
#SBATCH -J a40_TD3_BM
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt
#SBATCH -t 36:00:00
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -p allgroups
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40


cd $HOME/DT_thesis_BM/dt-sim
mkdir -p logs


export DISPLAY=:$((SLURM_JOB_ID % 100 + 100))
Xvfb $DISPLAY -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &

SIF_IMAGE="$HOME/DT_thesis_BM/dt-sim/duckie_rl.sif"

export LIBGL_ALWAYS_SOFTWARE=1
export PYGLET_HEADLESS=False
export GALLIUM_DRIVER=llvmpipe

# using my wand account
export WANDB_API_KEY=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
export WANDB_DIR=$PWD/wandb

srun singularity exec \
    -B .:/app \
    --pwd /app \
    --env DISPLAY=$DISPLAY \
    --env PYGLET_HEADLESS=False \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env GALLIUM_DRIVER=llvmpipe \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=$WANDB_DIR \
    $SIF_IMAGE \
    bash -c "export PYTHONPATH=\$PYTHONPATH:/app:/app/src && \
    python3 rl_bm/td3_continuous_action.py \
    --seed 123 \
    --env-id oval_loop \
    --total-timesteps 1000000 \
    --buffer-size 5000 \
    --learning-starts 5000 \
    --run-notes 'First test run of td3'"