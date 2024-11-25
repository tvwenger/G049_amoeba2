#!/bin/bash
#SBATCH --chdir="/home/twenger/G049_amoeba2"
#SBATCH --job-name="G049"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-888%175

# 891 spectra = 891 jobs of 1 spectra, limit to 175 tasks to limit resource usage
PER_JOB=1
NUM_SPEC=891
START_IDX=$(( $SLURM_ARRAY_TASK_ID * $PER_JOB ))
END_IDX=$(( ( $SLURM_ARRAY_TASK_ID + 1 ) * $PER_JOB ))

eval "$(conda shell.bash hook)"
conda activate amoeba2

for (( idx=$START_IDX; idx<$END_IDX; idx++ )); do
    if [ $idx -ge $NUM_SPEC ]; then
	    break
    fi
    
    # check if result already exists, then skip
    if [ -f "results/${idx}_amoeba2.pkl" ]; then
        echo "results/${idx}_amoeba2.pkl already exists!"
        continue
    fi

    # temporary pytensor compiledir
    tmpdir=`mktemp -d`
    echo "starting to analyze $idx"
    PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_G049.py $idx slurm
    rm -rf $tmpdir
done
