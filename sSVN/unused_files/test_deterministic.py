import numpy as np
import pytest
import itertools
# from models.gauss_analytic import gauss_analytic
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import SVI
import h5py
import deepdish as dd
import time
import os
import timeit
import functools
import logging
log = logging.getLogger(__name__)
np.seterr(over='raise')
class Test_deterministic_methods:
    def setup(self):
        self.rtol = 1e-25
        self.atol = 1e-25
        # Run settings
        # self.nIter = 150
        # self.nP = 200
        self.nIter = 2
        self.nP = 200
        self.method = 'SVGD'
        self.bw_selection = 'HE'
        # self.bw_selection = 'med'
        self.step_selection = 'linesearch'
        self.max_step = 1
        # Run setup
        self.model_1 = rosenbrock_analytic()
        self.stein_1 = SVI(model = self.model_1, nIterations=self.nIter, nParticles=self.nP, optimizeMethod=self.method)
        self.stein_1.bw_selection = self.bw_selection
        self.stein_1.stepsize_selection = self.step_selection
        self.stein_1.step_hyperparameter = self.max_step
        output_dict_1 = self.stein_1.constructMap()  # History file is produced by this method
        self.history_path_1 = output_dict_1['history_path_new']
        self.res_1 = dd.io.load(self.history_path_1)

        # self.stein_2 = SVI(model = self.model_2, nIterations=self.nIter, nParticles=self.nP, optimizeMethod=self.method) # This fails?
        # self.stein_2.bw_selection = self.bw_selection
        # self.stein_2.stepsize_selection = self.step_selection
        # self.stein_2.step_hyperparameter = self.max_step
        # time.sleep(3)
        # output_dict_2 = self.stein_2.constructMap()  # History file is produced by this method
        # time.sleep(3)
        # self.history_path_2 = output_dict_2['history_path_new']

        # self.res_2 = dd.io.load(self.history_path_2)

        # time.sleep(0.5)

    def test_bandwidth_determinism(self):
        nLoops = 2
        for i in range(nLoops):
            np.random.seed(int(time.time()))
            model_i = rosenbrock_analytic()
            stein_i = SVI(model=model_i, nIterations=self.nIter, nParticles=self.nP, optimizeMethod=self.method)
            stein_i.bw_selection = self.bw_selection
            stein_i.stepsize_selection = self.step_selection
            stein_i.step_hyperparameter = self.max_step
            dict = stein_i.constructMap()
            with h5py.File(dict['history_path_new'], 'r') as f:
                h1 = f['0']['h'][()]
                h2 = self.res_1['0']['h'][()]
                print(i)
                assert(np.allclose(h1, h2, rtol=self.rtol, atol=self.atol))
                assert(np.allclose(h1, 0.4309839989926324 , rtol=self.rtol, atol=self.atol))
            del model_i
            del stein_i
    # def test_bandwidth_determinism(self):
    #     for i in range(self.res_1['metadata']['total_num_iterations']):
    #         assert(np.allclose(self.res_1['%i' % i]['h'], self.res_2['%i' % i]['h'], rtol=self.rtol, atol=self.atol))
            # assert(np.array_equal(self.res_1['%i' % i]['h'], self.res_2['%i' % i]['h']))

    # def test_X_determinism(self):
    #     assert(np.allclose(self.res_1['final_updated_particles']['X'], self.res_2['final_updated_particles']['X'], rtol=self.rtol, atol=self.atol))
def main():
    a = Test_deterministic_methods()
    a.setup()
    a.test_bandwidth_determinism()
    # a.test_pushForward()

if __name__ is '__main__':
    main()