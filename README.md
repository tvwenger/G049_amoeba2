# G049_amoeba2
Run amoeba-2 on G049 data

## Installation
```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda create --name amoeba2 -c conda-forge pymc
conda activate amoeba2
pip install git+https://github.com/tvwenger/amoeba2.git
```