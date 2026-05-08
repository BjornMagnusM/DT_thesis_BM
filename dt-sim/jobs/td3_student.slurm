#!/bin/bash
#SBATCH -J Term3_500R_NoconstReward_l40_duckie_td3 
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt
#SBATCH -t 76:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem=16g
#SBATCH --nodelist=gpu7
#SBATCH --gres=gpu:l40s

cd $HOME/DT_thesis_BM/dt-sim
mkdir -p logs

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export LIBGL_ALWAYS_SOFTWARE=1
export PYGLET_HEADLESS=True
export GALLIUM_DRIVER=llvmpipe


export WANDB_API_KEY=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
export WANDB_DIR=$PWD/wandb


SIF_IMAGE="$HOME/DT_thesis_BM/dt-sim/duckie_rl.sif"


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
            --lap_termination \
            --time_optimal_reward \
            --run-notes 'added back penelty for angle 2M steps MIdlaneV2 Using Term3 with 500 for completed tile and no constreward'
    "