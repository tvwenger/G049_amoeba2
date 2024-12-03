"""fit_G034.py
Run amoeba2 on single G034 pixel
Trey V. Wenger - November 2024
"""

import sys
import pickle

import numpy as np

import pymc as pm
import bayes_spec
import amoeba2

from bayes_spec import SpecData, Optimize
from amoeba2 import AbsorptionModel


def main(idx, infile):
    print(f"Starting job on {infile}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"amoeba2 version: {amoeba2.__version__}")
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }

    # load data
    with open(infile, "rb") as f:
        datum = pickle.load(f)

    # load mol_data
    with open("mol_data.pkl", "rb") as f:
        mol_data = pickle.load(f)

    # get data
    data = {}
    for transition in ["1612", "1665", "1667", "1720"]:
        data[f"absorption_{transition}"] = SpecData(
            datum[f"velocity_{transition}"],
            datum[f"absorption_{transition}"],
            datum[f"absorption_rms_{transition}"],
            xlabel=r"$V_{\rm LSR}$ (km s$^{-1}$)",
            ylabel=r"$1 - e^{-\tau_{" + f"{transition}" + r"}}$",
        )

    try:
        # Initialize optimizer
        opt = Optimize(
            AbsorptionModel,  # model definition
            data,  # data dictionary
            max_n_clouds=10,  # maximum number of clouds
            baseline_degree=0,  # polynomial baseline degree
            mol_data=mol_data,  # molecular data
            seed=1234,  # random seed
            verbose=True,  # verbosity
        )

        # Define each model
        opt.add_priors(
            prior_tau=[0.1, 0.1],  # mean and width of log10(tau) prior
            prior_log10_depth=[0.0, 0.25],  # mean and width of log10(depth) prior (pc)
            prior_log10_Tkin=[2.0, 1.0],  # mean and width of log10(Tkin) prior (K)
            prior_velocity=[60.0, 10.0],  # mean and width of velocity prior (km/s)
            prior_log10_nth_fwhm_1pc=[0.2, 0.1],  # mean and width of non-thermal FWHM prior (km/s)
            prior_depth_nth_fwhm_power=[0.4, 0.1],  # mean and width of non-thermal FWHM exponent prior
            ordered=False,  # do not assume optically-thin
            mainline_pos_tau=True,  # force main line optical depths to be positive
        )
        opt.add_likelihood()

        # optimize
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.05,
            "learning_rate": 1e-2,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 1000,
            "draws": 1000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(bic_threshold=10.0, sample_kwargs=sample_kwargs, fit_kwargs=fit_kwargs, approx=False)

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {"bic": np.inf, "solutions": {}}
            for solution in model.solutions:
                # get BIC
                bic = model.bic(solution=solution)

                # get summary
                summary = pm.summary(model.trace[f"solution_{solution}"])

                # check convergence
                converged = summary["r_hat"].max() < 1.05

                if converged and bic < results[n_gauss]["bic"]:
                    results[n_gauss]["bic"] = bic

                # save posterior samples for un-normalized params (except baseline)
                data_vars = list(model.trace[f"solution_{solution}"].data_vars)
                data_vars = [data_var for data_var in data_vars if ("baseline" in data_var) or not ("norm" in data_var)]

                # only save posterior samples if converged
                results[n_gauss]["solutions"][solution] = {
                    "bic": bic,
                    "summary": summary,
                    "converged": converged,
                    "trace": (
                        model.trace[f"solution_{solution}"][data_vars].sel(draw=slice(None, None, 10))
                        if converged
                        else None
                    ),
                }
        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    idx = int(sys.argv[1])
    htc_type = sys.argv[2]

    if htc_type == "slurm":
        infile = f"G034/{idx}.pkl"
        outfile = f"G034_results/{idx}_amoeba2.pkl"
    else:
        infile = f"{idx}.pkl"
        outfile = f"{idx}_amoeba2.pkl"

    output = main(idx, infile)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    with open(outfile, "wb") as f:
        pickle.dump(output, f)
