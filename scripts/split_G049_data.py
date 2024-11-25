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

    # keep only those pixels where cont > n_sigma*sigma
    n_sigma = 3
    med = np.nanmedian(data["1612"]["cont"])
    rms = 1.4826 * np.nanmedian(np.abs(data["1612"]["cont"] - med))
    print(f"1612 MHz Continuum rms: {rms:.3f} Jy/beam")
    mask = data["1612"]["cont"] < (n_sigma * rms)
    print(f"Image contains {mask.size} pixels")
    print(f"Masking {mask.sum()} pixels with continuum < {n_sigma} sigma")

    # keep only those pixels where the absorption rms is <0.1
    for transition in transitions:
        new_mask = data[transition]["absorption_rms"] > 0.1
        print(f"Masking {new_mask.sum()} pixels with {transition} MHz rms > 0.1")
        mask += new_mask
    print(f"There are {(~mask).sum()} unmasked pixels")

    # split up spectra and save
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for file in glob.glob(os.path.join(outdir, "*.pkl")):
        os.remove(file)

    for idx, coord in enumerate(zip(*np.where(~mask))):
        print(idx, coord)
        if coord == (50, 50):
            print(f"Middle pixel is idx={idx}")
        datum = {"coord": coord}
        for transition in transitions:
            datum[f"velocity_{transition}"] = data[transition]["velocity"]
            datum[f"absorption_{transition}"] = data[transition]["absorption"][:, *coord]
            datum[f"absorption_rms_{transition}"] = data[transition]["absorption_rms"][*coord]
        with open(os.path.join(outdir, f"{idx}.pkl"), "wb") as f:
            pickle.dump(datum, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python split_G049_data.py <indir> <outdir>")
    else:
        main(indir=sys.argv[1], outdir=sys.argv[2])
