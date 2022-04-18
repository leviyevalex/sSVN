import numpy as np
import pytest
import itertools
# from models.gauss_analytic import gauss_analytic
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import SVI
import h5py
import deepdish as dd
import os
import timeit
import functools
class Test_auxilliary_methods:
    def setup(self):
        self.rtol = 1e-12
        self.atol = 1e-12
        self.model = rosenbrock_analytic()
        # self.stein = SVI(model = self.model, nIterations=100, nParticles=100, optimizeMethod='SVGD')
        # self.stein = SVI(model = self.model, nIterations=10, nParticles=10, optimizeMethod='SVGD') # This works
        self.stein = SVI(model = self.model, nIterations=11, nParticles=10, optimizeMethod='SVGD') # This fails?
        self.stein.step_hyperparameter = 0.01
        self.stein.bw_selection = 'HE'
        # self.stein.bw_selection = 'med'
        # self.stein.stepsize_selection = 'constant'
        self.stein.stepsize_selection = 'linesearch'
        output_dict = self.stein.constructMap()  # History file is produced by this method
        RUN_OUTPUT_DIR = output_dict['RUN_OUTPUT_DIR']
        self.history_path = output_dict['history_path_new']
        with h5py.File(self.history_path, 'r') as f:
            self.iterations = f['metadata'].attrs.__getitem__('total_num_iterations')
    def test_pushForward(self):
        res = dd.io.load(self.history_path)
        with h5py.File(self.history_path, 'r') as f:
            X0 = f['0']['X'][...] # initial particles
            X_algo = f['final_updated_particles']['X'][...]
        X_pushed = self.stein.pushForward(X0, self.history_path)
        assert(np.array_equal(X_algo, X_pushed))
        pass
def main():
    a = Test_auxilliary_methods()
    a.setup()
    a.test_pushForward()

if __name__ is '__main__':
    main()

