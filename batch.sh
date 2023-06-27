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
#SBATCH --time 12:00:00
#SBATCH --array=0-122

# there are 1228 pixels to fit
# That's 123 jobs in groups of 10
PER_JOB=10
START_IDX=$(( $SLURM_ARRAY_TASK_ID * $PER_JOB ))
END_IDX=$(( ( $SLURM_ARRAY_TASK_ID + 1 ) * $PER_JOB ))

eval "$(conda shell.bash hook)"
conda activate amoeba2

for (( idx=$START_IDX; idx<$END_IDX; idx++ )); do
    # check if data file does not exist, then skip
    if [ ! -f "amoeba_data/amoeba_data_$idx.pkl" ]; then
        continue
    fi

    # check if result already exists, then skip
    if [ -f "results/result_$idx.pkl" ]; then
        continue
    fi

    # temporary pytensor compiledir
    tmpdir=`mktemp -d`
    PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_G049_slurm.py $idx
    rm -rf $tmpdir
done
