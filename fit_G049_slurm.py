"""
fit_G049_slurm.py
Fit data with amoeba-2 using slurm
Trey Wenger - June 2023
"""

import sys
import pickle
import numpy as np
from amoeba2.amoeba import Amoeba


def main(idx):
    print(f"Starting job on idx = {idx}")

    # load data
    with open(f"amoeba_data/amoeba_data_{idx}.pkl", "rb") as f:
        datum = pickle.load(f)

    try:
        # Initialize amoeba
        amoeba = Amoeba(max_n_gauss=10, verbose=False, seed=1234)
        amoeba.set_prior("center", "normal", np.array([65.0, 2.0]))
        amoeba.set_prior("log10_fwhm", "normal", np.array([0.0, 0.33]))
        amoeba.set_prior("peak_tau", "normal", np.array([0.0, 0.1]))
        amoeba.add_likelihood("normal")

        # set data
        amoeba.set_data(datum["data"])

        # sample
        amoeba.fit_all(tune=500, draws=500, chains=4, cores=4)

        # save stats for all models
        results = {"coord": datum["coord"]}
        results[0] = {"bic": amoeba.models[1].null_bic()}
        for n_gauss in amoeba.models.keys():
            if amoeba.models[n_gauss].trace is not None:
                point_estimate = amoeba.models[n_gauss].point_estimate()
                lnlike = amoeba.models[n_gauss].lnlike_mean_point_estimate()
                bic = amoeba.models[n_gauss].bic()
                results[n_gauss] = {
                    "point_estimate": point_estimate,
                    "lnlike": lnlike,
                    "bic": bic,
                }
        return results

    except Exception as ex:
        return {"coord": datum["coord"], "exception": ex}


if __name__ == "__main__":
    output = main(int(sys.argv[1]))

    # save results
    fname = f"results/result_{sys.argv[1]}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(output, f)
