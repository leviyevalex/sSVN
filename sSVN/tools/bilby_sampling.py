#%% Load packages
from models.HRD import hybrid_rosenbrock
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import samplers
import bilby
import numpy as np
from collections import OrderedDict
from corner import corner
import os
from pathlib import Path

#%% Find root directory and load double banana ground truth file
root1 = Path(os.getcwd()).parent
root2 = os.getcwd()
if os.path.basename(root1) == 'stein_variational_newton':
    root = root1
else:
    root = root2
ground_truth_double_banana_path = os.path.join(root, 'double_banana_ground_truth.h5')

#%% 2D HRD
# n2 = 1
# n1 = 2
# model = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
# DoF = n2 * (n1 - 1) + 1

# %% 10D HRD
n2 = 3
n1 = 4
model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)
DoF = n2 * (n1 - 1) + 1

#%% Double Banana
# model_tmp = rosenbrock_analytic()
# stein = samplers(model=model_tmp, nIterations=1,nParticles=1, method='SVGD')
# DoF = 2

#%% Priors
priors = OrderedDict()
for d in range(DoF):
    # priors['%i' % d] = bilby.prior.Uniform(minimum=-20, maximum=20, name='%i' % d, latex_label='x_{%i}' % d) # 2D thin like
    priors['%i' % d] = bilby.prior.Uniform(minimum=-6, maximum=6, name='%i' % d, latex_label='x_{%i}' % d) # 10D thin like

#%% Likelihood wrapper for HRD
# class likelihood_wrapper(bilby.Likelihood):
#     def __init__(self):
#         self.dict = OrderedDict()
#         for d in range(DoF):
#             self.dict['%i' % d] = None
#         super().__init__(parameters=self.dict)
#         self.forward_evaluations = 0
#     def log_likelihood(self):
#         self.forward_evaluations += 1
#         vec = np.array(list(self.dict.values()))
#         return -1 * model.getMinusLogLikelihood_individual(vec)

#%% Likelihood wrapper for Double-Banana
class likelihood_wrapper(bilby.Likelihood):
    def __init__(self):
        self.dict = OrderedDict()
        for d in range(DoF):
            self.dict['%i' % d] = None
        super().__init__(parameters=self.dict)
        self.forward_evaluations = 0
    def log_likelihood(self):
        self.forward_evaluations += 1
        vec = np.array(list(self.dict.values()))
        return -1 * model.getMinusLogLikelihood(vec)
likelihood = likelihood_wrapper()
#%% Run sampler
# res = bilby.run_sampler(likelihood, priors, sampler="dynesty", nlive=1000)
outdir = os.path.join(root, 'nlive2000_bilby_results')
res = bilby.run_sampler(likelihood, priors, sampler="dynesty", nlive=2000, nact=50, outdir=outdir, check_point=False)
#%%
fig = corner(res.samples)
fig.show()
#%%
sSVN_path = os.path.join(root, 'outdir', '1641932744sSVN_identity_metric', 'output_data.h5')

#%% Cornerplots codd
import pandas as pd
df = pd.DataFrame(store1).assign(dataset='%s' % first_key)
df_key = pd.DataFrame(store2).assign(dataset='%s' % key)
df = pd.concat([df, df_key], ignore_index=True)
GT_df = pd.DataFrame(samples_GT[0:N]).assign(dataset='Truth')
df = pd.concat([df, GT_df], ignore_index=True)