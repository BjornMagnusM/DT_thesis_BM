#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --mail-user bjoernmagnus.myrhaug@studenti.unipd.it 

#SBATCH --output=test_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=2G


echo "=== START SMOKE TEST ==="

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gym-duckietown

echo "Python: $(which python)"
python --version

echo "Testing imports..."
python -c "
import numpy
import torch
import gymnasium
import duckietown_world
print('Core imports OK')
"

echo "Testing Duckietown env creation..."
python -c "
from gym_duckietown.envs import DuckietownEnv
env = DuckietownEnv(map_name='loop_empty', domain_rand=False, draw_render=False)
obs = env.reset()
print('Env reset OK')
"

echo "=== DONE ==="