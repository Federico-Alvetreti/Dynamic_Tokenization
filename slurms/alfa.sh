#!/bin/bash
#SBATCH -A IscrC_ELLIF-HE
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=DynTok_alfa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# load modules & env
module load cuda/12.2

# loop over alfa values
for alfa in 0.001  0.01  0.1; do
    echo "Running train.py with alfa=$alfa"
    python -m conda run -n dyntok python train.py method.parameters.alfa=$alfa

done

echo "All runs finished at $(date)"
