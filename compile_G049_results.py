"""compile_G049_results.py
Compile results from G049 data
Trey V. Wenger - August 2024
"""

import os
import sys
import glob

import pickle
import numpy as np

from astropy.io import fits
import matplotlib.pyplot as plt


def make_cube(results, wcs, label, filename, n_gauss="best", plot="summary", plot_key="velocity[0]", plot_stat="mean"):
    cube_size = (100, 100)
    bic_threshold = 10.0
    data = np.ones(cube_size) * np.nan

    for coord, result in results.items():
        # get best model
        bics = np.array([result["results"][i].get("bic", np.inf) for i in result["results"].keys()])
        n = n_gauss
        if n_gauss == "best":
            min_bic = bics.min()
            idx = np.where(bics < min_bic + bic_threshold)[0][0]
            n = list(result["results"].keys())[idx]
        if plot == "n_gauss":
            data[coord] = n
        elif plot == "bic":
            data[coord] = bics[n]
        elif plot in result["results"][n].keys():
            data[coord] = result["results"][n][plot][plot_stat][plot_key]
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
        bics = np.array([result["results"][i].get("bic", np.inf) for i in result["results"].keys()])
        min_bic = bics.min()
        idx = np.where(bics < min_bic + bic_threshold)[0][0]
        n = list(result["results"].keys())[idx]

        if "summary" in result["results"][n].keys():
            for i in range(n):
                mykeyx = keyx + f"[{i}]"
                if "]" in keyx:
                    mykeyx = keyx.replace("]", f", {i}]")
                mykeyy = keyy + f"[{i}]"
                if "]" in keyy:
                    mykeyy = keyy.replace("]", f", {i}]")
                datax.append(result["results"][n]["summary"]["mean"][mykeyx])
                errx.append(result["results"][n]["summary"]["sd"][mykeyx])
                datay.append(result["results"][n]["summary"]["mean"][mykeyy])
                erry.append(result["results"][n]["summary"]["sd"][mykeyy])
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
        basename = os.path.basename(datafile)
        resultfile = os.path.join(resultsdir, basename.replace("_data", "_results"))
        with open(resultfile, "rb") as f:
            result = pickle.load(f)
        results[data["coord"]] = result

    # Generate maps
    make_cube(results, wcs, "Min. BIC", "min_bic", n_gauss="best", plot="bic")
    make_cube(results, wcs, "Num. Clouds", "num_clouds", n_gauss="best", plot="n_gauss")
    for transition in ["1612", "1665", "1667", "1720"]:
        make_cube(
            results,
            wcs,
            r"rms$_{\tau, " + transition + r"}$",
            f"rms_{transition}",
            n_gauss="best",
            plot="summary",
            plot_key=f"rms_tau[{transition}]",
            plot_stat="mean",
        )
    fnames = ["fwhm_0", "velocity_0", "log10_N_0_0"]
    keys = ["fwhm[0]", "velocity[0]", "log10_N_0[0]"]
    labels = [r"$\Delta V$[0] (km s$^{-1}$)", r"$V_{\rm LSR}$[0] (km s$^{-1}$)", r"$\log_{10} N_0$[0] (cm$^{-2}$)"]
    for fname, key, label in zip(fnames, keys, labels):
        make_cube(
            results,
            wcs,
            label,
            fname,
            n_gauss="best",
            plot="summary",
            plot_key=key,
            plot_stat="mean",
        )
        make_cube(
            results,
            wcs,
            "Err. " + label,
            fname + "_sd",
            n_gauss="best",
            plot="summary",
            plot_key=key,
            plot_stat="sd",
        )

    # Generate cloud plots
    plot_cloud(
        results,
        "log10_N_0",
        r"$\log_{10} N_0$ (cm$^{-2}$)",
        "velocity",
        r"$V_{\rm LSR}$ (km s$^{-1}$)",
        "log10_N_0_vs_velocity",
    )
    plot_cloud(
        results,
        "fwhm",
        r"$\Delta V$ (km s$^{-1}$)",
        "velocity",
        r"$V_{\rm LSR}$ (km s$^{-1}$)",
        "fwhm_vs_velocity",
    )
    for transition in ["1612", "1665", "1667", "1720"]:
        plot_cloud(
            results,
            "velocity",
            r"$V_{\rm LSR}$ (km s$^{-1}$)",
            f"inv_Tex[{transition}]",
            r"$T_{\rm ex, " + transition + r"}$ (K$^{-1}$)",
            f"velocity_vs_inv_Tex_{transition}",
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("python compile_G049_results.py <wcs_file> <datadir> <resultsdir>")
    else:
        main(wcsfile=sys.argv[1], datadir=sys.argv[2], resultsdir=sys.argv[3])
