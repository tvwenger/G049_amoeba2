"""extract_G049_data.py
Extract data from G049 cubes
Trey V. Wenger - August 2024
"""

import os
import sys
import glob

import pickle
import numpy as np


def main(indir=".", outdir="."):
    transitions = ["1612", "1665", "1667", "1720"]
    data = {}
    for transition in transitions:
        with open(os.path.join(indir, f"data_{transition}.pkl"), "rb") as f:
            data[transition] = pickle.load(f)

    # keep only those pixels where all transitions have non-nan optical depths
    mask = np.zeros(data["1612"]["cont"].shape, dtype=bool)
    for transition in transitions:
        mask += np.any(np.isnan(data[transition]["tau"]), axis=0)

    # keep only those pixels where the optical depth rms is <0.1
    for transition in transitions:
        mask += data[transition]["tau_rms"] > 0.1
    print(f"There are {(~mask).sum()} unmasked pixels")

    # split up spectra and save
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for file in glob.glob(os.path.join(outdir, "*.pkl")):
        os.remove(file)

    for idx, coord in enumerate(zip(*np.where(~mask))):
        datum = {"coord": coord}
        for transition in transitions:
            datum[f"velocity_{transition}"] = data[transition]["velocity"]
            datum[f"tau_{transition}"] = data[transition]["tau"][:, *coord]
        with open(os.path.join(outdir, f"G049_data_{idx:05d}.pkl"), "wb") as f:
            pickle.dump(datum, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python split_G049_data.py <indir> <outdir>")
    else:
        main(indir=sys.argv[1], outdir=sys.argv[2])
