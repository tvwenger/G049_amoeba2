"""
fit_G049.py
Run amoeba-2 on the G049 data
Trey Wenger - June 2023
"""

from amoeba2.data import AmoebaData
from amoeba2.amoeba import Amoeba
from amoeba2.model import predict_tau_spectrum
import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy.io import fits
import multiprocessing as mp
import time
from spectral_cube import SpectralCube


def worker(datum):
    try:
        # Initialize amoeba
        amoeba = Amoeba(max_n_gauss=10, verbose=False, seed=1234)
        amoeba.set_prior("center", "normal", np.array([65.0, 2.0]))
        amoeba.set_prior("log10_fwhm", "normal", np.array([0.0, 0.33]))
        amoeba.set_prior("peak_tau", "normal", np.array([0.0, 0.05]))
        amoeba.add_likelihood("normal")

        # set data
        amoeba.set_data(datum["data"])

        # sample
        amoeba.fit_best(tune=500, draws=500, chains=4, cores=1)

        # save mean point estimate
        if amoeba.best_model is not None:
            point_estimate = amoeba.best_model.point_estimate()
            lnlike = amoeba.best_model.lnlike_mean_point_estimate()
            return {
                "coord": datum["coord"],
                "point_estimate": point_estimate,
                "lnlike": lnlike,
            }
        return None

    except Exception as ex:
        return {"coord": datum["coord"], "exception": ex}


def main():
    transitions = ["1612", "1665", "1667", "1720"]
    fnames = [
        "/data/vla/G049.205-0.343.spw36.I.channel.clean.pbcor.image.fits",  # 1612
        "/data/vla/G049.205-0.343.spw35.I.channel.clean.pbcor.image.fits",  # 1665
        "/data/vla/G049.205-0.343.spw6.I.channel.clean.pbcor.image.fits",  # 1667
        "/data/vla/G049.205-0.343.spw5.I.channel.clean.pbcor.image.fits",  # 1720
    ]

    # storage for WCS
    wcs = None

    data = {}
    common_beam = None
    for transition, fname in zip(transitions, fnames):
        cube = SpectralCube.read(fname)
        cube.allow_huge_operations = True

        # chop off edge channels
        # most of the emission seems to come from 300-700
        cube = cube[100:-100, :, :]

        # smoothing to common beam
        if common_beam is None:
            # worst beam will be at 1612. The channels are basically
            # the same
            common_beam = cube.beams[150]
        else:
            cube = cube.convolve_to(common_beam)

        # keep only pixel range 300 to 400
        cube = cube[:, 300:400, 300:400]

        # continuum estimate
        cont = cube.median(axis=0)

        # save WCS
        if wcs is None:
            wcs = cont.wcs

        # continuum subtract
        contsub = cube - cont

        # optical depth
        tau = contsub / cont.filled_data[:]
        tau = tau.unitless_filled_data[:]

        # estimate optical depth rms over line-free channels
        tau_line_free = np.concatenate([tau[:200], tau[600:]])
        med = np.median(tau_line_free, axis=0)
        rms = 1.4826 * np.median(np.abs(tau_line_free - med), axis=0)

        # save
        data[transition] = {
            "tau": tau,
            "cont": cont,
            "tau_rms": rms,
            "velocity": cube.spectral_axis.to("km/s").value,
        }

    # save data to disk
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    # start here if you need to restart to save time
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # mask pixels where any transition has spectral rms > 0.10
    mask = np.any(
        [data[transition]["tau_rms"] > 0.10 for transition in transitions], axis=0
    )

    # for the unmasked pixels, remove a baseline and store the coordinates and
    # an AmoebaData object
    amoeba_data = []
    for coord in zip(*np.where(~mask)):
        datum = {"coord": coord, "data": AmoebaData()}
        for transition in transitions:
            spec = data[transition]["tau"][:, *coord]
            datum["tau_original"] = spec

            # identify line-free channels with 2-sigma threshold
            line_free = np.where(
                np.abs(spec) < 2.0 * data[transition]["tau_rms"][*coord]
            )[0]

            # fifth order baseline fit
            coeff = np.polyfit(line_free, spec[line_free], deg=5)
            fit = np.polyval(coeff, np.arange(len(spec)))

            # keep only relevant channel range
            keep = (data[transition]["velocity"] > 50.0) * (
                data[transition]["velocity"] < 80.0
            )
            datum["data"].set_spectrum(
                transition,
                data[transition]["velocity"][keep],
                (spec - fit)[keep],
                data[transition]["tau_rms"][*coord],
            )
        amoeba_data.append(datum)

    # save data to disk
    with open("amoeba_data.pkl", "wb") as f:
        pickle.dump(amoeba_data, f)

    # dump each data file to disk
    for i, datum in enumerate(amoeba_data):
        with open(f"/data/vla/amoeba_G049/amoeba_data_{i}.pkl", "wb") as f:
            pickle.dump(datum, f)

    # plot average of spectra weighted by rms
    fig, ax = plt.subplots()
    for transition in transitions:
        spectra = np.array(
            [foo["data"].spectra[transition].spectrum for foo in amoeba_data]
        )
        weights = (
            1.0
            / np.array([foo["data"].spectra[transition].rms for foo in amoeba_data])
            ** 2.0
        )
        avg_spec = np.average(spectra, weights=weights, axis=0)
        ax.plot(
            amoeba_data[0]["data"].spectra[transition].velocity,
            avg_spec,
            linestyle="-",
            label=transition,
        )
    ax.set_xlabel(r"LSR Velocity (km s$^{-1}$)")
    ax.set_ylabel(r"Optical Depth")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("avg_spec.pdf", bbox_inches="tight")
    plt.close(fig)

    # start here if you need to save time
    with open("amoeba_data.pkl", "rb") as f:
        amoeba_data = pickle.load(f)

    # storage for results
    results = {}

    # parallelize over pixels because amoeba-2 is slow...
    start = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, output in enumerate(pool.imap_unordered(worker, amoeba_data)):
            # save result
            if output is not None:
                results[output["coord"]] = output

            # print status
            now = time.time()
            elapsed = now - start
            time_per = elapsed / (i + 1)
            num_left = len(amoeba_data) - i - 1
            time_left = time_per * num_left
            print("=====================================")
            print(f"Completed: {i+1}/{len(amoeba_data)}")
            print(f"Time elapsed: {elapsed/60.0:.2f} min")
            print(f"Iteration: {time_per/60.0:.2f} min")
            print(f"Remaining: {time_left/60.0:.2f} min")
            print("=====================================")

    # save results to disk! should probably do this at every iteration
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    # generate FITS header
    header = wcs.to_header()
    for key, val in common_beam.to_header_keywords().items():
        header[key] = val

    # storage for data
    lnlike = np.ones(data["1612"]["cont"].shape) * np.nan
    n_gauss = np.ones(data["1612"]["cont"].shape, dtype=int) * np.nan
    for coord in results.keys():
        if "exception" in results[coord].keys():
            print(f"{coord} had an exception: {results[coord]['exception']}")
        else:
            lnlike[*coord] = results[coord]["lnlike"]
            n_gauss[*coord] = len(results[coord]["point_estimate"]["center"]["mean"])

    # plot single spectrum with results
    coord = (42, 63)
    fig, axes = plt.subplots(4, sharex=True)
    for transition, ax in zip(transitions, axes):
        # plot data
        ax.plot(data[transition]["velocity"], data[transition]["tau"][:, *coord], "k-")

        # plot fit
        fit = predict_tau_spectrum(
            data["1612"]["velocity"][:, None],
            results[coord]["point_estimate"][f"peak_tau_{transition}"]["mean"],
            results[coord]["point_estimate"]["center"]["mean"],
            10.0 ** np.array(results[coord]["point_estimate"]["log10_fwhm"]["mean"]),
        )
        ax.plot(data["1612"]["velocity"], fit, "r-", linewidth=2)
        ax.set_ylabel(r"$\tau_{" + transition + "}$")
    ax.set_xlabel(r"LSR Velocity (km s$^{-1}$)")
    fig.show()
