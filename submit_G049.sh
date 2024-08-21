#!/bin/bash
#SBATCH --chdir="/home/twenger/G049_amoeba2"
#SBATCH --job-name="G049"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --time 24:00:00
#SBATCH --array=0-888%175

# 889 spectra = 889 jobs of 1 spectra, limit to 175 tasks to limit resource usage
PER_JOB=1
NUM_SPEC=889
START_IDX=$(( $SLURM_ARRAY_TASK_ID * $PER_JOB ))
END_IDX=$(( ( $SLURM_ARRAY_TASK_ID + 1 ) * $PER_JOB ))

eval "$(conda shell.bash hook)"
conda activate amoeba2

for (( idx=$START_IDX; idx<$END_IDX; idx++ )); do
    if [ $idx -ge $NUM_SPEC ]; then
	    break
    fi
    
    fmtidx=$(printf "%05d" $idx)
    # check if result already exists, then skip
    if [ -f "results/G049_results_$fmtidx.pkl" ]; then
        echo "results/G049_results_$fmtidx.pkl already exists!"
        continue
    fi

    # temporary pytensor compiledir
    tmpdir=`mktemp -d`
    echo "starting to analyze $idx"
    PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_G049.py $idx
    rm -rf $tmpdir
done
