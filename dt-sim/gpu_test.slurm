#!/bin/bash
#SBATCH -J duckie_td3
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt
#SBATCH -t 36:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40

cd $HOME/DT_thesis_BM/dt-sim
mkdir -p logs

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# -------------------------
# 1. Start virtual display
# -------------------------
export DISPLAY=:99
Xvfb $DISPLAY -screen 0 1024x768x24 -ac +extension GLX +render -noreset > xvfb.log 2>&1 &
sleep 2

# -------------------------
# 2. Rendering setup (CRITICAL)
# -------------------------
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export PYGLET_HEADLESS=False
export PYOPENGL_PLATFORM=glx

# -------------------------
# 3. WandB
# -------------------------
export WANDB_API_KEY=YOUR_KEY_HERE
export WANDB_DIR=$PWD/wandb

# -------------------------
# 4. Container
# -------------------------
SIF_IMAGE="$HOME/DT_thesis_BM/dt-sim/duckie_rl.sif"

# -------------------------
# 5. Run inside container
# -------------------------
srun singularity exec \
    -B .:/app \
    --pwd /app \
    --env DISPLAY=$DISPLAY \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env GALLIUM_DRIVER=llvmpipe \
    --env PYGLET_HEADLESS=False \
    --env PYOPENGL_PLATFORM=glx \
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
            --run-notes 'TD3 test run'
    "