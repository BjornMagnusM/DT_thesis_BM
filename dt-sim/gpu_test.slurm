#!/bin/bash
#SBATCH -J l40s duckie_td3 
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt
#SBATCH -t 36:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s

cd $HOME/DT_thesis_BM/dt-sim
mkdir -p logs

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# -------------------------
# Rendering → CPU ONLY
# -------------------------
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export PYOPENGL_PLATFORM=egl
export PYGLET_HEADLESS=True
export EGL_DEVICE_ID=none

# -------------------------
# WandB
# -------------------------
export WANDB_API_KEY=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
export WANDB_DIR=$PWD/wandb

# -------------------------
# Container
# -------------------------
SIF_IMAGE="$HOME/DT_thesis_BM/dt-sim/duckie_rl.sif"

# -------------------------
# Run
# -------------------------
srun singularity exec --nv \
    -B .:/app \
    --pwd /app \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env GALLIUM_DRIVER=llvmpipe \
    --env PYOPENGL_PLATFORM=egl \
    --env PYGLET_HEADLESS=True \
    --env EGL_DEVICE_ID=none \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=$WANDB_DIR \
    $SIF_IMAGE \
    bash -c "
        export PYTHONPATH=\$PYTHONPATH:/app:/app/src && \
        python3 rl_bm/td3_continuous_action.py \
            --seed 123 \
            --env-id oval_loop \
            --total-timesteps 1000000 \
            --buffer-size 5000 \
            --learning-starts 5000 \
            --run-notes 'TD3 GPU training + CPU rendering'
    "