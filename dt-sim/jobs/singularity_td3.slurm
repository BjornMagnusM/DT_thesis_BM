#!/bin/bash
#SBATCH --job-name=LoopRGB_Best 
#SBATCH --output=output/log_%j.out
#SBATCH -e output/log_%j.err
#SBATCH --time=76:00:00
#SBATCH --partition=pgpu_most
#SBATCH --account=dei_most
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4



cd $HOME/DT_thesis_BM/dt-sim
mkdir -p output

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export LIBGL_ALWAYS_SOFTWARE=1
export PYGLET_HEADLESS=True
export GALLIUM_DRIVER=llvmpipe


export WANDB_API_KEY=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
export WANDB_DIR=$PWD/wandb


SIF_IMAGE="$HOME/duckie_rl.sif"


srun singularity exec --nv \
    -B .:/app \
    --pwd /app \
    $SIF_IMAGE \
    bash -c "
        export PYTHONPATH=\$PYTHONPATH:/app:/app/src && \
        python3 rl_bm/td3_continuous_action.py \
            --seed 2 \
            --env-id oval_loop \
            --total-timesteps 2000000 \
            --buffer-size 10000 \
            --learning-starts 10000 \
            --time_optimal_reward \
            --lap_termination \
            --distortion \
            --dynamics_rand \
            --camera_rand \
            --run-notes "TD3 no dis save best RGB New loop"
    "