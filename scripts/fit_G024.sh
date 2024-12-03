#!/bin/bash

source activate
conda activate amoeba2

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_G024.py $1 condor
rm -rf $tmpdir