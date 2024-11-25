# G049_amoeba2
Run `amoeba2` on G049 data with SLURM or Condor.

## Installation
```bash
git clone git@github.com:tvwenger/G049_amoeba2.git
cd G049_amoeba2
mkdir data/
mkdir results/
mkdir logs/
conda env create -f environment.yaml
conda activate amoeba2
```

## First steps
First, we smooth data cubes to a common beam resolution and calculate the absorption spectra.
```
python extract_G049_data.py /path/to/G049_fits_images/
```
This script outputs new pickle files in `/path/to/G049_fits_images/` like `data_1612.pkl` which contain the trimmed absorption (`1-exp(-tau)`) cube, the continuum image, the absorption rms estimate, and the velocity axis definition for each transition. It also saves the WCS object to `wcs.pkl`.

Next, we split up the pixels into individual data files for parallel processing.
```
python split_G049_data.py /path/to/G049_fits_images/ /path/to/individual_data_pickle_files/
```
The script selects a subset of pixels (those with sufficient signal-to-noise) and dumps each pixel into an individual pickle file in `/path/to/individual_data_pickle_files/`. Each pickle file contains the velocity axes and absorption (`1-exp(-tau)`) spectra for each transition.

## Parallel processing with SLURM
The script `fit_G049.py` runs `amoeba2`'s optimization algorithm on a single pickle file. With SLURM, the data are assumed to be in `data/` and the results go in `results/` as individual pickle files.
```
python fit_G049.py <idx> slurm
```

The script `submit_G049.sh` is a SLURM script that we use to analyze each pixel in parallel. SLURM logs are written to `logs/`.
```
sbatch submit_G049.sh
```

## Parallel processing with Condor
An `amoeba2` docker container is provided via the `Dockerfile`:
```
docker build -t tvwenger/amoeba2:v1.1.0 .
docker push tvwenger/amoeba2:v1.1.0
```

Eventually, this docker container [should be converted into an Apptainer image](https://chtc.cs.wisc.edu/uw-research-computing/htc-docker-to-apptainer):
```
condor_submit -i build.sub
apptainer build amoeba2-v1.1.0.sif docker://tvwenger/amoeba2:v1.1.0
mv amoeba2-v1.1.0.sif /staging/tvwenger/
```

The script `fit_G049.py` runs `amoeba2`'s optimization algorithm on a single pickle file. With Condor, the data are assumed to be in the local directory as individual pickle files.
```
python fit_G049.py <idx> condor
```

The script `G049.sub` is a Condor script that we use to analyze each pixel in parallel. It handles the copying of data and results to `data/` and `results/`, respectively. Condor logs are written to `logs/`.
```
condor_submit limit=1000 G049.sub
```

Whenever we switch to Apptainer, then we'll have to update the submission script:
```
# Provide HTCondor with the name of your .sif file and universe information
requirements = (HasCHTCStaging == true)
container_image = file:///staging/twenger2/amoeba2-v1.1.0.sif
```

## Analysis
Finally, the script `compile_G049_results.py` compiles the results into some FITS images and graphics.
```
mkdir fits/
mkdir figures/
python compile_G049_results.py /path/to/wcs.pkl /path/to/individual_data_pickle_files/ /path/to/individual_results_pickle_files/
```
This script outputs several images to `fits/` and `figures/`. 