"""compile_G049_results.py
Compile results from G049 data
Trey V. Wenger - August 2024
"""

import os
import sys
import glob
import copy

import pickle
import numpy as np

from astropy.io import fits
import matplotlib.pyplot as plt


def get_best_model(result, bic_threshold=10.0):
    """Determine the best amoeba2 result, and return the best solution."""
    # keep only best model
    best_bic = np.inf
    best_n_gauss = 0
    best_solution = 0
    best_num_solutions = 0

    # check all models
    for n_gauss in result["results"].keys():
        this_bic = np.inf
        this_solution = None
        this_num_solutions = 0
        if "bic" in result["results"][n_gauss]:
            this_bic = result["results"][n_gauss]["bic"]

        # check all solutions
        if "solutions" in result["results"][n_gauss].keys():
            this_num_solutions = len(result["results"][n_gauss]["solutions"])
            for solution in result["results"][n_gauss]["solutions"].keys():
                converged = result["results"][n_gauss]["solutions"][solution]["converged"]
                bic = result["results"][n_gauss]["solutions"][solution]["bic"]
                if converged and bic <= this_bic:
                    this_bic = bic
                    this_solution = solution

        # compare to current best model
        if np.isinf(best_bic) or this_bic < (best_bic - bic_threshold):
            best_bic = this_bic
            best_n_gauss = n_gauss
            best_solution = this_solution
            best_num_solutions = this_num_solutions

    # return best model
    compiled_result = copy.deepcopy(result)
    if best_n_gauss in result["results"].keys():
        compiled_result["results"] = result["results"][best_n_gauss]
        if "solutions" in result["results"][best_n_gauss].keys() and best_solution is not None:
            compiled_result["results"] = result["results"][best_n_gauss]["solutions"][best_solution]
            return compiled_result, best_bic, best_n_gauss, best_num_solutions

    # no good model
    return None, best_bic, best_n_gauss, best_num_solutions


def make_cube(results, wcs, label, filename, plot="summary", plot_key="velocity[0]", plot_stat="mean"):
    cube_size = (100, 100)
    bic_threshold = 10.0
    data = np.ones(cube_size) * np.nan

    for coord, result in results.items():
        # get best model
        result, best_bic, best_n_gauss, best_num_solutions = get_best_model(result, bic_threshold=bic_threshold)
        if result is None:
            data[coord] = np.nan
        elif plot == "n_gauss":
            data[coord] = best_n_gauss
        elif plot == "bic":
            data[coord] = best_bic
        elif plot == "n_solutions":
            data[coord] = best_num_solutions
        elif plot in result["results"].keys():
            data[coord] = result["results"][plot][plot_stat][plot_key]
        else:
            data[coord] = np.nan

    # Write FITS cube
    fits.writeto(os.path.join("fits", f"{filename}.fits"), data, wcs.to_header(), overwrite=True)

    # Generate figure
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(projection=wcs)
    cax = ax.imshow(data, origin="lower")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Decl. (J2000)")
    cbar = fig.colorbar(cax)
    cbar.set_label(label)
    fig.savefig(os.path.join("figures", f"{filename}.png"))
    plt.close(fig)


def plot_cloud(results, keyx, labelx, keyy, labely, filename):
    bic_threshold = 10.0
    datax = []
    errx = []
    datay = []
    erry = []
    clouds = []
    for _, result in results.items():
        # get best model
        result, best_bic, best_n_gauss, best_num_solutions = get_best_model(result, bic_threshold=bic_threshold)
        if result is None:
            continue

        if "summary" in result["results"].keys():
            for i in range(best_n_gauss):
                mykeyx = keyx + f"[{i}]"
                if "]" in keyx:
                    mykeyx = keyx.replace("]", f", {i}]")
                mykeyy = keyy + f"[{i}]"
                if "]" in keyy:
                    mykeyy = keyy.replace("]", f", {i}]")
                datax.append(result["results"]["summary"]["mean"][mykeyx])
                errx.append(result["results"]["summary"]["sd"][mykeyx])
                datay.append(result["results"]["summary"]["mean"][mykeyy])
                erry.append(result["results"]["summary"]["sd"][mykeyy])
                clouds.append(i)

    fig, ax = plt.subplots(layout="constrained")
    clouds_norm = np.array(clouds) / np.max(clouds)
    c = plt.get_cmap("viridis")(clouds_norm)
    cax = ax.scatter(datax, datay, c=clouds, marker=".", alpha=0.75)
    ax.errorbar(datax, datay, xerr=errx, yerr=erry, marker="none", ls="none", ecolor=c, elinewidth=0.5, alpha=0.75)
    cbar = fig.colorbar(cax)
    cbar.set_label("Cloud Number")
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    fig.savefig(os.path.join("figures", f"{filename}.png"))
    plt.close(fig)


def main(wcsfile="wcs.pkl", datadir=".", resultsdir="."):
    # Load WCS object
    with open(wcsfile, "rb") as f:
        wcs = pickle.load(f)

    # Get coordinates from data, load results
    datafiles = glob.glob(os.path.join(datadir, "*.pkl"))
    results = {}
    for datafile in datafiles:
        with open(datafile, "rb") as f:
            data = pickle.load(f)
        idx = os.path.basename(datafile).replace(".pkl", "")
        resultfile = os.path.join(resultsdir, f"{idx}_amoeba2.pkl")
        with open(resultfile, "rb") as f:
            result = pickle.load(f)
        results[data["coord"]] = result

    # Generate maps
    make_cube(results, wcs, "Min. BIC", "min_bic", plot="bic")
    make_cube(results, wcs, "Num. Clouds", "num_clouds", plot="n_gauss")
    make_cube(results, wcs, "Num. Solutions", "num_solutions", plot="n_solutions")
    fnames = ["fwhm_0", "velocity_0", "tau_1665_0"]
    keys = ["fwhm[0]", "velocity[0]", "tau_1665[0]"]
    labels = [r"$\Delta V$[0] (km s$^{-1}$)", r"$V_{\rm LSR}$[0] (km s$^{-1}$)", r"$\tau_{1665}$[0]"]
    for fname, key, label in zip(fnames, keys, labels):
        make_cube(
            results,
            wcs,
            label,
            fname,
            plot="summary",
            plot_key=key,
            plot_stat="mean",
        )
        make_cube(
            results,
            wcs,
            "Err. " + label,
            fname + "_sd",
            plot="summary",
            plot_key=key,
            plot_stat="sd",
        )

    # Generate cloud plots
    plot_cloud(
        results,
        "fwhm",
        r"$\Delta V$ (km s$^{-1}$)",
        "velocity",
        r"$V_{\rm LSR}$ (km s$^{-1}$)",
        "fwhm_vs_velocity",
    )
    transitions = ["1612", "1665", "1667", "1720"]
    for i, transition in enumerate(transitions):
        plot_cloud(
            results,
            "velocity",
            r"$V_{\rm LSR}$ (km s$^{-1}$)",
            f"tau_{transition}",
            r"$\tau_{" + transition + r"}$",
            f"velocity_vs_tau_{transition}",
        )
        for other_transition in transitions[i + 1 :]:
            plot_cloud(
                results,
                f"tau_{transition}",
                r"$\tau_{" + transition + r"}$",
                f"tau_{other_transition}",
                r"$\tau_{" + other_transition + r"}$",
                f"tau_{transition}_vs_tau_{other_transition}",
            )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python compile_G049_results.py <wcs_file> <datadir> <resultsdir>")
    else:
        main(wcsfile=sys.argv[1], datadir=sys.argv[2], resultsdir=sys.argv[3])
