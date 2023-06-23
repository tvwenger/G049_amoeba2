#!/bin/bash
#SBATCH --chdir="/home/twenger/G049_amoeba2"
#SBATCH --job-name="g049_amoeba2"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --nodes=1-20
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --time 2-00:00:00
#SBATCH --array 0-890

# N.B. Update --array above to match number of pixels to be fit

eval "$(conda shell.bash hook)"
conda activate amoeba2

python fit_G049_slurm.py $SLURM_ARRAY_TASK_ID