# G049_amoeba2
Run `amoeba2` on G049 data with SLURM.

## Installation
```bash
git clone https://github.com/tvwenger/G049_amoeba2
cd G049_amoeba2
mkdir data/
mkdir results/
mkdir logs/
conda env create -f environment.yml
conda activate amoeba2
```

## Usage

First, we smooth data cubes to a common beam resolution and calculate the optical depth.
```
python extract_G049_data.py /path/to/data
```
This script outputs new pickle files in `/path/to/data` like `data_1612.pkl` which contain the trimmed optical depth cube, the continuum image, the optical depth rms estimate, and the velocity axis definition for each transition.

Next, we split up the pixels into individual data files for parallel processing.
```
python split_G049_data.py /path/to/data /path/to/output
```
The script selects a subset of pixels (those with reliable optical depth measurements) and dumps each pixel into an individual pickle file in `/path/to/output`. Each pickle file contains the velocity axes and optical depth spectra for each transition.

The script `fit_G049.py` runs `amoeba2`'s optimization algorithm on a single pickle file, assumed to exist in a subdirectory `data/` and stores the result in `results/` as individual pickle files. The script `submit_G049.sh` is a SLURM script that we use to analyze each pixel in parallel. SLURM logs are written to `logs/`.
