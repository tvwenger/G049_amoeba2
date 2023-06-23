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
#SBATCH --array 1-45

# there are 890 pixels to fit
NUM_JOBS=890

# each "task" will fit 20 pixels
PER_TASK=20

# so there must be at least 45 tasks (--array)

START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
END_NUM=$(( $END_NUM < $NUM_JOBS ? $END_NUM : $NUM_JOBS ))
echo Task $SLURM_ARRAY_TASK_ID running $START_NUM to $END_NUM

eval "$(conda shell.bash hook)"
conda activate amoeba2

for (( i=$START_NUM; i<=END_NUM; i++ )); do
    python fit_G049_slurm.py $i
done

