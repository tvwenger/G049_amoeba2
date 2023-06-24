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
#SBATCH --time 02:00:00

# check if this is already a task
if [[ "$SLURM_ARRAY_TASK_ID" == "" ]]; then
    # if not, then we need to figure out what pixels are left to fit

    # get pixel ids of data
    idxs=()
    for datafile in amoeba_data/*.pkl; do
        idxs+=(`echo "$datafile" | awk -F'[_.]' '{ print $4 }'`)
    done

    # remove completed pixels
    for resultfile in results/*.pkl; do
        idx=`echo "$resultfile" | awk -F'[_.]' '{ print $2 }'`
        for i in "${!idxs[@]}"; do
            if [[ ${idxs[i]} = $idx ]]; then
                unset 'idxs[i]'
            fi
        done
    done

    # Relaunch this script as an array
    array=`echo ${idxs[@]} | tr ' ' ','`
    exec sbatch --array="$array" $0
fi

eval "$(conda shell.bash hook)"
conda activate amoeba2

# temporary pytensor compiledir
tmpdir=`mktemp -d`
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_G049_slurm.py $SLURM_ARRAY_TASK_ID
rm -rf $tmpdir