from plotz import *
from math import sin, pi
import numpy as np
import os
import h5py

# output_data_path = os.path.join(output_dir, 'output_data_new.h5')


with Plot("moment_convergence_plot") as p:
    file = '../plots/double-banana-sSVN-500.h5'
    with h5py.File(file, 'r') as hf:
        keys = [int(l) for l in list(hf.keys())[:-2]]
        keys.sort()
        iters_performed = keys[-1]
        mean_x = np.zeros(iters_performed)
        mean_y = np.zeros(iters_performed)
        mean = np.zeros(iters_performed)
        cov_x = np.zeros(iters_performed)
        cov_y = np.zeros(iters_performed)
        for l in range(iters_performed):
            print(l)
            X = hf['%i' % l]['X'][()]
            mean = np.mean(X, axis=0)
            # mean[l] = np.mean(X)
            cov = np.cov(X.T)
            mean_x[l] = mean[0]
            mean_y[l] = mean[1]
            cov_x[l] = cov[0, 0]
            cov_y[l] = cov[1, 1]
    # p.title = r"My first \texttt{PlotZ} plot"
    p.x.label = "Number of iterations $l$"
    # p.y.label = "$y$"
    # p.y.label_rotate = True
    # p.style.dashed()
    p.style.colormap("monochrome")
    # p.y.label_rotate = True
    # p.plot(Function(sin, samples=50, range=(0, pi)),
    #        title=r"$\sin(x)$")
    # print('plotting first')
    # p.plot(mean_x, title=r"\mu_x")
    # print('plotting second')
    np.savetxt('output.csv', mean_x, delimiter=',')
    p.plot(DataFile("output.csv", sep=",", comment="#"), title=r"PDSA(6)")
    # p.plot(DataFile("output.csv"))
    # p.plot(cov_x, title=r"\sigma_x")
    # p.plot(cov_y, title=r"\sigma_y")
    # p.legend("north east")
