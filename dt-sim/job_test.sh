#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --mail-user bjoernmagnus.myrhaug@studenti.unipd.it 

#SBATCH --output=test_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=2G

echo "=== START ==="

echo "User: $(whoami)"
echo "Node: $(hostname)"

echo "Checking python..."
which python
python --version

echo "Checking conda..."
which conda

echo "PATH:"
echo $PATH

echo "=== DONE ==="