#%%
import os
import h5py
import matplotlib.pyplot as plt
import deepdish as dd
#%%
cwd = os.getcwd()
folder = '/1613269546 HE this one is awesome'
filename = '/rosenbrock_proper_nP_1000_200_1613269546.h5'
metadata = '/drivers/outdir' + folder + filename # 2d rosen good
file = cwd + metadata
results = dd.io.load(file)[200]['X']
ground_truth_path = cwd + '/models/ground_truths/double_rosenbrock/3k_samples_rosen_test.h5'
ground_truth = dd.io.load(ground_truth_path)
# ground_truth_path = cwd + '/models/ground_truths/double_rosenbrock/170k.h5'
# ground_truth = dd.io.load(ground_truth_path).T
#%%
# background = '/drivers/outdir/' + 'hybrid_rosenbrock_-2.50_2.50.h5'
background = '/drivers/outdir/' + 'rosenbrock_proper_-2.50_2.50.h5'
background_path = cwd + background
background_data = dd.io.load(background_path)
X = background_data['X']
Y = background_data['Y']
Z = background_data['Z']
#%%
fig, ax = plt.subplots(figsize = (10, 10))
cp = ax.contour(X, Y, Z, 7, colors='black', alpha=0.1)
ax.set_facecolor('#F5FEFF')
ax.scatter(results[:,0], results[:, 1], marker=".", color='#51AEFF', s=8)
fig.show()