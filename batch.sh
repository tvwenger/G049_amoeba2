#!/bin/bash
#SBATCH --chdir="/home/twenger/G049_amoeba2"
#SBATCH --job-name="g049_amoeba2"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --time 00:10:00
#SBATCH --array 0-890

# N.B. update --array to match number of pixels to fit

eval "$(conda shell.bash hook)"
conda activate amoeba2

python fit_G049_slurm.py $SLURM_ARRAY_TASK_ID
