#!/usr/bin/env python
import numpy as np
import bilby
from bilby.core.prior import Uniform
import pandas as pd
import tqdm

np.random.seed(1234)
sigma = 1
Nresults = 100
Nsamples = 1000
Nparameters = 10
Nruns = 1

priors = {f"x{jj}": Uniform(-1, 1, f"x{jj}") for jj in range(Nparameters)}


for x in range(Nruns):
    results = []
    for ii in tqdm.tqdm(range(Nresults)):
        posterior = dict()
        injections = dict()
        for key, prior in priors.items():
            sim_val = prior.sample()
            rec_val = sim_val + np.random.normal(0, sigma)
            posterior[key] = np.random.normal(rec_val, sigma, Nsamples)
            injections[key] = sim_val

        posterior = pd.DataFrame(dict(posterior))
        result = bilby.result.Result(
            label="test",
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=injections.keys(),
            priors=priors)
        results.append(result)

    bilby.result.make_pp_plot(results, filename=f"run{x}_90CI",
                              confidence_interval=0.9)
    bilby.result.make_pp_plot(results, filename=f"run{x}_3sigma",
                              confidence_interval=[0.68, 0.95, 0.997])