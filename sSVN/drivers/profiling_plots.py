#%%
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import timeit
import time
import opt_einsum as oe
import functools

#%%
nparticles = np.array([3000, 2000, 1000, 500, 50])

# memory = np.log(np.array([1981.5, 913, 272, 111.6, 62.2]))

time_per_iteration = np.log(np.array([3371.85, 935.35, 115, 14.56, 7.16]))

# plt.plot(nparticles, memory)
plt.plot(nparticles, time_per_iteration)
plt.show()
# # Memory in MiB, key is # particles
# dict_memory = {'3000' : 1981.5, 913}
#
# # Time per iteration in seconds, key is # particles
# dict_time_iteration = {'3000' : 3371.85}
#%%
A = np.random.rand(500, 500)
B = np.random.rand(500, 500)
#%%
start = time.time()
for i in range(100):
    np.einsum('ij, jl -> il', A, B)
end = time.time()
print(end - start)
#%%
start = time.time()
for i in range(100):
    np.einsum('ij, ki -> jk', A, B)
end = time.time()
print(end - start)
#%%
start = time.time()
for i in range(100):
    A @ B
    # np.einsum('ij, ki -> jk', A, B)
end = time.time()
print(end - start)
# t_contract = timeit.Timer((np.einsum('ij, jl -> il', A, B)))
# t_contract = timeit.Timer(functools.partial(np.einsum('ij, jl -> il', A, B)))
#%%
# a = A @ B
# b =np.einsum('ij, jk -> ik', A, B)
# start = time.time()
# ((np.einsum('xy, xz, xbd -> yzbd', kernel, kernel, GN_Hmlpt) + np.einsum('xyb, xzd -> yzbd', gradKernel, gradKernel)) / self.nParticles).swapaxes(1, 2).reshape(dim, dim)
# 'xy, xz, xbd -> yzbd'v
#%%
n = 5000
d = 2
dim = n * d
contract_a = oe.contract_expression("xy, xz, xbd -> yzbd", (n, n), (n, n), (n, d, d))
contract_b = oe.contract_expression("xyb, xzd -> yzbd", (n, n, d), (n, n, d))
contract_term_mgJ = oe.contract_expression('mn, mo -> no', (n, n),(n, d))
kern = np.random.rand(n, n)
gmlpt = np.random.rand(n, d)
hess = np.random.rand(n, d, d)
gkern = np.random.rand(n, n, d)
hbar = np.random.rand(n, n, d, d)

#%%
start = time.time()
for i in range(1):
    # res_naive = np.einsum("xy, xz, xbd -> yzbd", kern, kern, hess) + np.einsum("xyb, xzd -> yzbd", gkern, gkern)
    # res_contract = contract_a(kern, kern, hess) + contract_b(gkern, gkern)
    # np.einsum('mn, mo -> no', kern, gmlpt)
    contract_term_mgJ(kern, gmlpt)
    # hbar.swapaxes(1, 2).reshape(dim, dim)
    ###

end = time.time()
print(end - start)
