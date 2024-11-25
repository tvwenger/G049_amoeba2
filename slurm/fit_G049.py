"""fit_G049.py
Run amoeba2 on single G049 pixel
Trey V. Wenger - August 2024
"""

import os
import sys
import pickle

import numpy as np

import pymc as pm
import bayes_spec
import amoeba2

from bayes_spec import SpecData, Optimize
from amoeba2 import TauModel


def main(idx):
    print(f"Starting job on idx = {idx}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"amoeba2 version: {amoeba2.__version__}")
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }

    # load data
    with open(f"data/G049_data_{idx:05d}.pkl", "rb") as f:
        datum = pickle.load(f)

    # load mol_data
    with open("mol_data.pkl", "rb") as f:
        mol_data = pickle.load(f)

    # get data
    data = {}
    for transition in ["1612", "1665", "1667", "1720"]:
        # estimate rms
        med = np.median(datum[f"tau_{transition}"])
        rms = 1.4826 * np.median(np.abs(datum[f"tau_{transition}"] - med))
        data[f"tau_{transition}"] = SpecData(
            datum[f"velocity_{transition}"],
            datum[f"tau_{transition}"],
            rms,
            xlabel=r"$V_{\rm LSR}$ (km s$^{-1}$)",
            ylabel=r"$\tau_{" + f"{transition}" + r"}$",
        )

    try:
        # Initialize optimizer
        opt = Optimize(
            TauModel,  # model definition
            data,  # data dictionary
            max_n_clouds=5,  # maximum number of clouds
            baseline_degree=0,  # polynomial baseline degree
            mol_data=mol_data,  # molecular data
            seed=1234,  # random seed
            verbose=True,  # verbosity
        )

        # Define each model
        opt.add_priors(
            prior_log10_N_0=[13.0, 1.0],  # mean and width of log10(N_u) prior (cm-2)
            prior_inv_Tex=[0.1, 1.0],  # mean and width of 1/Tex prior (K-1)
            prior_fwhm=1.0,  # mode of FWHM line width prior (km/s)
            prior_velocity=[65.0, 5.0],  # mean and width of velocity prior (km/s)
            prior_rms_tau=0.05,  # width of optical depth rms prior
            ordered=False,  # do not assume optically-thin
        )
        opt.add_likelihood()

        # optimize
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        }
        sample_kwargs = {
            "chains": 4,
            "cores": 4,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(bic_threshold=10.0, sample_kwargs=sample_kwargs, fit_kwargs=fit_kwargs, approx=True)

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {}
            if len(model.solutions) > 1:
                results[n_gauss]["exception"] = "multiple solutions"
            elif len(model.solutions) == 1:
                results[n_gauss]["bic"] = model.bic(solution=0)
                results[n_gauss]["summary"] = pm.summary(model.trace.solution_0)
            else:
                results[n_gauss]["exception"] = "no solution"
        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    idx = int(sys.argv[1])
    output = main(idx)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    outdirname = "results"
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)
    fname = f"{outdirname}/G049_results_{idx:05d}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(output, f)
