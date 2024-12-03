"""extract_G024_G034_data.py
Extract data from G024 and G034 cubes
Trey V. Wenger - December 2024
"""

import os
import sys
import warnings

import pickle
import numpy as np

from spectral_cube import SpectralCube


def main(indir="."):
    datasets = {
        "G024": [
            os.path.join(
                indir, "G024.471+0.492.spw5.I.channel.clean.pbcor.image.smoothed.2.fits"
            ),  # 1612
            os.path.join(
                indir, "G024.471+0.492.spw6.I.channel.clean.pbcor.image.smoothed.2.fits"
            ),  # 1665
            os.path.join(
                indir,
                "G024.471+0.492.spw35.I.channel.clean.pbcor.image.smoothed.2.fits",
            ),  # 1667
            os.path.join(
                indir,
                "G024.471+0.492.spw36.I.channel.clean.pbcor.image.smoothed.2.fits",
            ),  # 1720
        ],
        "G034": [
            os.path.join(
                indir, "G034.256+0.145.spw5.I.channel.clean.pbcor.image.smoothed.2.fits"
            ),  # 1612
            os.path.join(
                indir, "G034.256+0.145.spw6.I.channel.clean.pbcor.image.smoothed.2.fits"
            ),  # 1665
            os.path.join(
                indir,
                "G034.256+0.145.spw35.I.channel.clean.pbcor.image.smoothed.2.fits",
            ),  # 1667
            os.path.join(
                indir,
                "G034.256+0.145.spw36.I.channel.clean.pbcor.image.smoothed.2.fits",
            ),  # 1720
        ],
    }
    transitions = ["1612", "1665", "1667", "1720"]

    for source in datasets.keys():
        wcs = None

        for transition, fname in zip(transitions, datasets[source]):
            print(f"Processing: {source} {transition} data")
            cube = SpectralCube.read(fname)
            cube.allow_huge_operations = True

            # chop off edge channels
            # keep only center of image
            if source == "G024":
                cube = cube[100:-100, 200:300, 200:300]
            elif source == "G034":
                cube = cube[100:-100, 300:400, 300:400]
            cube.write(
                os.path.join(indir, f"{source}_{transition}_cube.fits"),
                format="fits",
                overwrite=True,
            )

            # continuum estimate
            cont = cube.median(axis=0)
            cont.write(
                os.path.join(indir, f"{source}_{transition}_cont.fits"),
                format="fits",
                overwrite=True,
            )

            # save WCS
            if wcs is None:
                wcs = cont.wcs

            # absorption 1-exp(-tau)
            with warnings.catch_warnings(action="ignore"):
                absorption = 1.0 - cube._data / cont._data
            absorption_cube = SpectralCube(data=absorption, wcs=cube.wcs)
            absorption_cube.write(
                os.path.join(indir, f"{source}_{transition}_absorption.fits"),
                format="fits",
                overwrite=True,
            )

            # estimate optical depth rms over line-free channels
            absorption_line_free = np.concatenate([absorption[:100], absorption[-100:]])
            med = np.nanmedian(absorption_line_free, axis=0)
            rms = 1.4826 * np.nanmedian(np.abs(absorption_line_free - med), axis=0)

            # save
            data = {
                "absorption": absorption,
                "cont": cont,
                "absorption_rms": rms,
                "velocity": cube.spectral_axis.to("km/s").value,
            }

            with open(os.path.join(indir, f"{source}_{transition}.pkl"), "wb") as f:
                pickle.dump(data, f)

            # save rms
            cont.data = rms
            cont.write(
                os.path.join(indir, f"{source}_{transition}_rms.fits"),
                format="fits",
                overwrite=True,
            )

        # save WCS
        with open(os.path.join(indir, f"{source}_wcs.pkl"), "wb") as f:
            pickle.dump(wcs, f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python extract_G024_G034_data.py <indir>")
    else:
        main(indir=sys.argv[1])
