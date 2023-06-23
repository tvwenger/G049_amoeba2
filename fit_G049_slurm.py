"""
fit_G049_slurm.py
Fit data with amoeba-2 using slurm
Trey Wenger - June 2023
"""

import sys
import pickle
import numpy as np
from amoeba2.model import AmoebaTauModel
from amoeba2.amoeba import Amoeba


def main(idx):
    print(f"Starting job on idx = {idx}")

    # load data
    with open("amoeba_data_20230622.pkl", "rb") as f:
        amoeba_data = pickle.load(f)
    datum = amoeba_data[idx]

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
        """
        # run one model to debug
        # Initialize the model
        model = AmoebaTauModel(
            n_gauss=4,  # number of components
            seed=1234,  # random number generator seed
            verbose=True,
        )
        model.set_prior("center", "normal", np.array([65.0, 2.0]))
        model.set_prior("log10_fwhm", "normal", np.array([0.0, 0.33]))
        model.set_prior("peak_tau", "normal", np.array([0.0, 0.05]))
        model.add_likelihood("normal")
        model.set_data(datum["data"])
        model.fit(tune=500, draws=500, chains=4, cores=1)
        return {
            "coord": datum["coord"],
            "point_estimate": model.point_estimate(),
            "lnlike": model.lnlike_mean_point_estimate(),
        }
        """

    except Exception as ex:
        return {"coord": datum["coord"], "exception": ex}


if __name__ == "__main__":
    output = main(int(sys.argv[1]))

    # save results
    fname = f"results/result_{sys.argv[1]}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(output, f)
