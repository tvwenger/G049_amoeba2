"""compile_results.py
Compile results from amoeba2 analysis
Trey V. Wenger - August 2024
Updates: December 2024
"""

import os
import sys
import glob
import copy

import pickle
import numpy as np
import pandas as pd

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
                converged = result["results"][n_gauss]["solutions"][solution][
                    "converged"
                ]
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
        if (
            "solutions" in result["results"][best_n_gauss].keys()
            and best_solution is not None
        ):
            compiled_result["results"] = result["results"][best_n_gauss]["solutions"][
                best_solution
            ]

            # order components by velocity
            velocity = np.array(
                [
                    compiled_result["results"]["summary"]["mean"][f"velocity[{i}]"]
                    for i in range(best_n_gauss)
                ]
            )
            compiled_result["results"]["order"] = np.argsort(velocity)
            return compiled_result, best_bic, best_n_gauss, best_num_solutions

    # no good model
    return None, best_bic, best_n_gauss, best_num_solutions


def make_cube(
    results,
    wcs,
    label,
    filename,
    plot="summary",
    plot_key="velocity",
    cloud_idx=None,
    baseline_idx=None,
    plot_stat="mean",
    outdir=".",
):
    cube_size = (100, 100)
    data = np.ones(cube_size) * np.nan

    for coord, result in results.items():
        if result is None:
            data[coord] = np.nan
        elif plot == "n_gauss":
            data[coord] = result["best_n_gauss"]
        elif plot == "bic":
            data[coord] = result["best_bic"]
        elif plot == "n_solutions":
            data[coord] = result["best_num_solutions"]
        elif (
            plot in result["results"].keys()
            and cloud_idx is None
            and baseline_idx is None
        ):
            data[coord] = result["results"][plot][plot_stat][f"{plot_key}"]
        elif plot in result["results"].keys() and baseline_idx is not None:
            data[coord] = result["results"][plot][plot_stat][
                f"{plot_key}[{baseline_idx}]"
            ]
        elif (
            plot in result["results"].keys() and cloud_idx in result["results"]["order"]
        ):
            data[coord] = result["results"][plot][plot_stat][
                f"{plot_key}[{result['results']['order'][cloud_idx]}]"
            ]
        else:
            data[coord] = np.nan

    # Write FITS cube
    fits.writeto(
        os.path.join(outdir, f"{filename}.fits"), data, wcs.to_header(), overwrite=True
    )

    # Generate figure
    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(projection=wcs)
    cax = ax.imshow(data, origin="lower")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Decl. (J2000)")
    cbar = fig.colorbar(cax)
    cbar.set_label(label)
    fig.savefig(os.path.join(outdir, f"{filename}.png"))
    plt.close(fig)


def main(wcsfile="wcs.pkl", datadir=".", source="G049", bic_threshold=10.0, outdir="."):
    # amoeba2 parameters
    baseline_degree = 0
    max_clouds = 10

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Load WCS object
    with open(wcsfile, "rb") as f:
        wcs = pickle.load(f)

    # Get coordinates from data, load results
    datafiles = glob.glob(os.path.join(datadir, source, "*.pkl"))
    results = {}
    for datafile in datafiles:
        with open(datafile, "rb") as f:
            data = pickle.load(f)
        idx = os.path.basename(datafile).replace(".pkl", "")
        resultfile = os.path.join(datadir, f"{source}_results", f"{idx}_amoeba2.pkl")
        with open(resultfile, "rb") as f:
            result = pickle.load(f)
        results[data["coord"]] = result

    # Get best model
    for coord, result in results.items():
        result, best_bic, best_n_gauss, best_num_solutions = get_best_model(
            result, bic_threshold=bic_threshold
        )
        if result is not None:
            result["best_bic"] = best_bic
            result["best_n_gauss"] = best_n_gauss
            result["best_num_solutions"] = best_num_solutions
        results[coord] = result

    # Write cloud parameters to table
    cloud_params = [
        "tau_1612",
        "tau_1665",
        "tau_1667",
        "tau_1720",
        "log10_depth",
        "log10_Tkin",
        "velocity",
        "fwhm",
    ]
    columns = ["x", "y"]
    for param in cloud_params:
        columns += [f"{param}_mean", f"{param}_sd"]
    table = pd.DataFrame(columns=columns)
    for coord, result in results.items():
        if result is None:
            continue
        for cloud_idx in range(max_clouds):
            if cloud_idx not in result["results"]["order"]:
                continue
            row = {
                "x": [coord[0]],
                "y": [coord[1]],
            }
            for param in cloud_params:
                row[f"{param}_mean"] = [
                    result["results"]["summary"]["mean"][
                        f"{param}[{result['results']['order'][cloud_idx]}]"
                    ]
                ]
                row[f"{param}_sd"] = [
                    result["results"]["summary"]["sd"][
                        f"{param}[{result['results']['order'][cloud_idx]}]"
                    ]
                ]
            row = pd.DataFrame.from_dict(row)
            table = pd.concat([table, row], ignore_index=True)
    table.to_csv(os.path.join(outdir, f"{source}_cloud_params.csv"), index=False)

    # Generate maps of model results
    make_cube(results, wcs, "BIC", f"{source}_bic", plot="bic", outdir=outdir)
    make_cube(
        results,
        wcs,
        "Number of Clouds",
        f"{source}_num_clouds",
        plot="n_gauss",
        outdir=outdir,
    )
    make_cube(
        results,
        wcs,
        "Number of Solutions",
        f"{source}_num_solutions",
        plot="n_solutions",
        outdir=outdir,
    )

    # Generate maps of hyperparameters
    fnames = [
        f"{source}_log10_nth_fwhm_1pc",
        f"{source}_depth_nth_fwhm_power",
    ]
    keys = [
        "log10_nth_fwhm_1pc",
        "depth_nth_fwhm_power",
    ]
    labels = [
        r"$\Delta V_{\rm 1 pc}$ (km s$^{-1}$)",
        r"$\alpha$",
    ]
    for fname, key, label in zip(fnames, keys, labels):
        make_cube(
            results,
            wcs,
            label,
            fname,
            plot="summary",
            plot_key=key,
            plot_stat="mean",
            outdir=outdir,
        )
        make_cube(
            results,
            wcs,
            "Err. " + label,
            fname + "_sd",
            plot="summary",
            plot_key=key,
            plot_stat="sd",
            outdir=outdir,
        )

    # Generate maps of baseline parameters
    for baseline_idx in range(baseline_degree + 1):
        for transition in ["1612", "1665", "1667", "1720"]:
            label = r"$\beta_{" + transition + r"}$" + f"[{baseline_idx}]"
            fname = f"{source}_baseline_{transition}_{baseline_idx}"
            key = f"baseline_absorption_{transition}_norm"
            make_cube(
                results,
                wcs,
                label,
                fname,
                plot="summary",
                plot_key=key,
                baseline_idx=baseline_idx,
                plot_stat="mean",
                outdir=outdir,
            )
            make_cube(
                results,
                wcs,
                "Err. " + label,
                fname + "_sd",
                plot="summary",
                plot_key=key,
                baseline_idx=baseline_idx,
                plot_stat="sd",
                outdir=outdir,
            )

    # Generate maps of cloud parameters
    for cloud_idx in range(max_clouds):
        fnames = [
            f"{source}_fwhm_{cloud_idx}",
            f"{source}_velocity_{cloud_idx}",
            f"{source}_tau_1612_{cloud_idx}",
            f"{source}_tau_1665_{cloud_idx}",
            f"{source}_tau_1667_{cloud_idx}",
            f"{source}_tau_1720_{cloud_idx}",
            f"{source}_log10_depth_{cloud_idx}",
            f"{source}_log10_Tkin_{cloud_idx}",
        ]
        keys = [
            "fwhm",
            "velocity",
            "tau_1612",
            "tau_1665",
            "tau_1667",
            "tau_1720",
            "log10_depth",
            "log10_Tkin",
        ]
        labels = [
            r"$\Delta V$[" + f"{cloud_idx}" + r"] (km s$^{-1}$)",
            r"$V_{\rm LSR}$[" + f"{cloud_idx}" + r"] (km s$^{-1}$)",
            r"$\tau_{1612}$[" + f"{cloud_idx}" + r"]",
            r"$\tau_{1665}$[" + f"{cloud_idx}" + r"]",
            r"$\tau_{1667}$[" + f"{cloud_idx}" + r"]",
            r"$\tau_{1720}$[" + f"{cloud_idx}" + r"]",
            r"$\log_{10} d$[" + f"{cloud_idx}" + r"]",
            r"$\log_{10} T_K$[" + f"{cloud_idx}" + r"]",
        ]
        for fname, key, label in zip(fnames, keys, labels):
            make_cube(
                results,
                wcs,
                label,
                fname,
                plot="summary",
                plot_key=key,
                cloud_idx=cloud_idx,
                plot_stat="mean",
                outdir=outdir,
            )
            make_cube(
                results,
                wcs,
                "Err. " + label,
                fname + "_sd",
                plot="summary",
                plot_key=key,
                cloud_idx=cloud_idx,
                plot_stat="sd",
                outdir=outdir,
            )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "python compile_results.py <wcs_file> <datadir> <source> <bic_threshold> <outdir>"
        )
        print("e.g. python compile_results.py wcs.pkl /data/ G049 10.0 /data/bic10/")
    else:
        main(
            wcsfile=sys.argv[1],
            datadir=sys.argv[2],
            source=sys.argv[3],
            bic_threshold=float(sys.argv[4]),
            outdir=sys.argv[5],
        )
