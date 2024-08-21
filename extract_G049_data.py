"""extract_G049_data.py
Extract data from G049 cubes
Trey V. Wenger - August 2024
"""

import os
import sys
import warnings

import pickle
import numpy as np

from spectral_cube import SpectralCube


def main(indir="."):
    transitions = ["1612", "1665", "1667", "1720"]
    fnames = [
        os.path.join(indir, "G049.205-0.343.spw36.I.channel.clean.pbcor.image.fits"),  # 1612
        os.path.join(indir, "G049.205-0.343.spw35.I.channel.clean.pbcor.image.fits"),  # 1665
        os.path.join(indir, "G049.205-0.343.spw6.I.channel.clean.pbcor.image.fits"),  # 1667
        os.path.join(indir, "G049.205-0.343.spw5.I.channel.clean.pbcor.image.fits"),  # 1720
    ]

    # storage for WCS
    wcs = None

    common_beam = None
    for transition, fname in zip(transitions, fnames):
        print(f"Processing: {transition} data")
        cube = SpectralCube.read(fname)
        cube.allow_huge_operations = True

        # chop off edge channels
        # most of the emission seems to come from 300-700
        # keep only center of image
        cube = cube[100:-100, 300:400, 300:400]

        # smoothing to common beam
        if common_beam is None:
            # worst beam will be at 1612. The channels are basically the same
            common_beam = cube.beams[150]
        else:
            cube = cube.convolve_to(common_beam)

        # continuum estimate
        cont = cube.median(axis=0)

        # save WCS
        if wcs is None:
            wcs = cont.wcs

        # optical depth
        with warnings.catch_warnings(action="ignore"):
            tau = -np.log(cube / cont)

        # estimate optical depth rms over line-free channels
        tau_line_free = np.concatenate([tau[:200], tau[600:]])
        med = np.nanmedian(tau_line_free, axis=0)
        rms = 1.4826 * np.nanmedian(np.abs(tau_line_free - med), axis=0)

        # save
        data = {
            "tau": tau,
            "cont": cont,
            "tau_rms": rms,
            "velocity": cube.spectral_axis.to("km/s").value,
        }

        with open(os.path.join(indir, f"data_{transition}.pkl"), "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python extract_G049_data.py <indir>")
    else:
        main(indir=sys.argv[1])
