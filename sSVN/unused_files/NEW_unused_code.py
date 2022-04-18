# def test_SVGD_Diffusion(self):
#     for m, n in itertools.product(range(self.X.shape[0]), range(self.T.shape[0])):
#         assert(np.allclose(self.DK_nndd[m,n], self.kernel[m,n] / self.nParticles * np.eye(self.DoF), rtol=self.rtol, atol=self.atol))

# def test_divD(self):
#     test_a = self.stein.getDivergenceDiffusionContribution(self.gradKernel, self.grad_hij_BD, self.HBD_inv_K, self.K_HBD_inv).flatten()
#     # K = self.formK()
#     # H_INV_BD = self.makeFullMatrixFromBlockDiagonal(self.HBD_inv)
#     # test_transpose = K @ H_INV_BD
#     gradK = self.gradK_single_parameter_def()
#     gradH = self.formGradH_bar_BD()
#     HBD_INV_K = self.stein.reshapeNNDDtoNDND(self.HBD_inv_K)
#     K_HBD_INV = self.stein.reshapeNNDDtoNDND(self.K_HBD_inv)
#     a = np.einsum('abe, be -> a', gradK, HBD_INV_K)
#     b = np.einsum('aA, ABe, Be -> a', K_HBD_INV, gradH, HBD_INV_K)
#     c = np.einsum('ac, cee -> a', K_HBD_INV, gradK)
#     test_b = self.nParticles * (a - b + c)
#     assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)
#     pass

# def form_gradK_manually(self, X, h, metric):
#     # X = X.reshape(self.nParticles, self.DoF)
#     dim = self.nParticles * self.DoF
#     res = np.zeros((dim, dim, dim))
#     deltas = self.stein.getDeltas(X, X)
#     metricDeltas = self.stein.getMetricDeltas(metric, deltas)
#     deltaMetricDeltas = self.stein.getDeltasMetricDeltas(deltas, metricDeltas)
#     kx = self.stein.getKernelPiecewise(h, deltaMetricDeltas)
#     gkx = self.stein.getGradKernelPiecewise(h, kx, metricDeltas)
#     #############################################################
#     # Get the gradient of augmented kernel matrix
#     #############################################################
#     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
#         zeros = np.zeros((self.nParticles, self.DoF))
#         zeros[n] = copy.deepcopy(gkx[m, n])
#         zeros = zeros.flatten()
#         block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
#         res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
#     res /= -1 * self.nParticles
#     divK_a = np.mean(gkx, axis=0).flatten()
#     divK_b = np.einsum('acc -> a', res)
#     try:
#         assert np.allclose(divK_a, divK_b, rtol=self.rtol, atol=self.atol)
#         return res
#     except:
#         return None

# def formGradH_bar(self):
#     dim = self.nParticles * self.DoF
#     res = np.zeros((dim, dim, dim))
#     #############################################################
#     # Helper function to get the blocks for the matrix
#     #############################################################
#     def makeBlock(m, n):
#         block = np.zeros((self.DoF, self.DoF, dim))
#         for d, b in itertools.product(range(self.DoF), range(self.DoF)):
#             entry = np.zeros((self.nParticles, self.DoF))
#             entry[m] = copy.deepcopy(self.gradH[m, n, d, b, :])
#             entry = entry.flatten()
#             block[d, b, :] = entry
#         return block
#     #############################################################
#     # Construct augmented grad H matrix explicitly
#     #############################################################
#     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
#         block = makeBlock(m, n)
#         res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
#     return res
# def test_gradHBD(self):
#     for z, i, j, k in itertools.product(range(self.nParticles), range(self.DoF), range(self.DoF), range(self.DoF)):
#         res = np.mean(-2 * self.GNHmlpt_new[:, i, j] * self.kernel[:,z] * self.gradKernel[:,z,k] - self.Hesskx[:, z, i, k] * self.gradKernel[:,z,j] - self.gradKernel[:,z,i] * self.Hesskx[:,z,j,k])
#         assert np.allclose(res, self.grad_hij_BD[z,i,j,k], rtol=self.rtol, atol=self.atol)


# Scratchwork for gradK_action_mnbd
# First slot gradient action scratchwork
# first_entry_test = np.einsum('mb, mb -> ', self.gradKernel[0, :], Bnndd[:, 0, 0]) / self.nParticles
# assert np.allclose(first_entry_test, vec_naive_a[0], rtol=self.rtol, atol=self.atol)
# first_block_test = np.einsum('mb, mdb -> d', self.gradKernel[0, :], Bnndd[:, 0]) / self.nParticles
# assert np.allclose(first_block_test, vec_naive_a[0:self.DoF], rtol=self.rtol, atol=self.atol)
# test_one = np.einsum('mnb, nmdb -> md', self.gradKernel, Bnndd) / self.nParticles
# assert np.allclose(test_one.flatten(), vec_naive_a, rtol=self.rtol, atol=self.atol)
# # Second slot gradient action result //////////////////////////////////////////////
# test_two = np.einsum('nbd, nmd -> mb', Bndd, self.gradKernel) / self.nParticles
# assert np.allclose(test_two.flatten(), vec_naive_b, rtol=self.rtol, atol=self.atol)
#
# # Test chain rule action
# res_c = self.gradK_single_parameter_def()
# vec_naive_c = np.einsum('abc, bc -> a', res_c, Bndnd)
# test_chain_rule = test_one + test_two
# assert np.allclose(test_chain_rule.flatten(), vec_naive_c, rtol=self.rtol, atol=self.atol)

# gradHBD_action_mnbd scracthwork ///////////
# Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
# res = self.formGradH_bar_BD()
# test_a = np.einsum('abc, bc -> a', res, Bndnd)
#
# # First entry in test_a
# a = np.trace(self.grad_hij_BD[0, 0, :, :] @ Bnndd[0, 0].T)
# assert np.allclose(test_a[0], a, rtol=1e-14, atol=1e-14)
#
# # First entry in test_a: rewritten with einsum
# b = np.einsum('db, db -> ', self.grad_hij_BD[0, 0, :, :], Bnndd[0, 0])
# assert np.allclose(test_a[0], b, rtol=1e-14, atol=1e-14)
#
# # First block in test_a
# c = np.einsum('adb, db -> a', self.grad_hij_BD[0, :, :, :], Bnndd[0, 0])
# assert np.allclose(test_a[0:self.DoF], c, rtol=1e-14, atol=1e-14)
#
# # All blocks in test_a
# d = np.einsum('nadb, nndb -> na', self.grad_hij_BD, Bnndd).flatten()
# assert np.allclose(test_b, d, rtol=1e-14, atol=1e-14)
# pass

# TESTING grad_hij_BD with both slot derivative
# First entry in first block
# np.einsum('nij, nij -> ', self.grad_hij_BD[0,:,0], Bnndd[0])
# First block
# np.einsum('nikj, nij -> k', self.grad_hij_BD[0], Bnndd[0])
# Whole thing
# test_a = np.einsum('mnikj, mnij -> mk', self.grad_hij_BD, Bnndd).flatten()
# For debugging purposes
# #######################
# # Checks out (OUTDATED)
# #######################
# def formGradH_bar_BD(self):
#     res = np.zeros((self.dim, self.dim, self.dim))
#     # Helper function to get the blocks for the matrix
#     def makeBlock(m):
#         block = np.zeros((self.DoF, self.DoF, self.dim))
#         for d, b in itertools.product(range(self.DoF), range(self.DoF)):
#             entry = np.zeros((self.nParticles, self.DoF))
#             entry[m] = copy.deepcopy(self.grad_hij_BD[m, d, b, :])
#             entry = entry.flatten()
#             block[d, b, :] = entry
#         return block
#     # Construct augmented grad H matrix explicitly
#     for m in range(self.nParticles):
#         block = makeBlock(m)
#         res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1), :] = block
#     return res

# def formGradH_BD_second_slot(self): #(OUTDATED)
#     res = np.zeros((self.dim, self.dim, self.dim))
#     # Helper function to get the blocks for the matrix
#     def makeBlock(m):
#         block = np.zeros((self.DoF, self.DoF, self.dim))
#         for i, j in itertools.product(range(self.DoF), range(self.DoF)):
#             vec = self.grad_hij_second_slot[m,i,:,j].flatten()
#             block[i, j] = vec
#         return block
#     for m in range(self.nParticles):
#         res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1), :] = makeBlock(m)
#     return res

# (5) //////////////////////
# def test_gradHBD_action_mnbd(self):
#     # Compare by forming action directly
#     Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
#     test_a = self.stein.gradHBD_action_mnbd(self.grad_hij_BD, Bnndd)
#     res = self.formGradH_bar_BD()
#     Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
#     test_b = np.einsum('abc, bc -> a', res, Bndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-14, atol=1e-14)

# def test_gradHBD_action_second_slot_mnbd(self):
#     # Make sure numerical agrees with analytic
#     gradH_first = self.formGradH_bar_BD()
#     gradH_second = self.formGradH_BD_second_slot()
#     total_gradH = gradH_first + gradH_second
#     gradH_numerical = self.test_numerical_gradient_h_ij_BD()
#
#     Bnndd = np.random.rand(self.nParticles, self.nParticles, self.DoF, self.DoF)
#     Bndnd = self.stein.reshapeNNDDtoNDND(Bnndd)
#     test_b = np.einsum('abc, bc -> a', total_gradH, Bndnd)

##########################################################
# Stochastic SVN Block Diagonal: Calculation methods (4)
##########################################################
# (1) ////////////////////////////////////////////
# grad_hij_BD_new(self, Hmlpt, kx, gkx, hesskx):
# (2) ///////////////////////////////////////////////////////
# getSVN_Direction(self, gkx, D_SVN, gmlpt_new, K_HBD_inv):
# (3) ///////////////////////////////////////////////////
# getSVN_stochastic_correction(self, kx, HBD_inv_sqrt):
# (4) ////////////////////////////////////////////////////////////////////////////
# getSVN_deterministic_correction(self, gkx, grad_hij_BD, HBD_inv_K, K_HBD_inv):
# def gradK_single_parameter_def(self):
#     # This is the complete gradient of the diffusion matrix for SVGD
#     dim = self.nParticles * self.DoF
#     res = np.zeros((dim, dim, dim))
#     # Get the gradient of augmented kernel matrix
#     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
#         zeros = np.zeros((self.nParticles, self.DoF))
#         zeros[m] = copy.deepcopy(self.gradKernel[m, n])
#         zeros[n] = copy.deepcopy(-1 * self.gradKernel[m, n])
#         zeros = zeros.flatten()
#         block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
#         res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
#     res /= self.nParticles
#     rep_a = np.mean(self.gradKernel, axis=0).flatten()
#     rep_b = np.einsum('acc -> a', res)
#     try:
#         assert np.allclose(rep_a, rep_b, rtol=self.rtol, atol=self.atol)
#         return res
#     except:
#         return None


# def grad_hij_BD_second_slot(self, kx, gkx, Hesskx, GN_Hmlpt, grad_GN_Hmlpt):
#     # For second slot
#     a = np.einsum('mn, mne, nij -> minje', 2 * kx, gkx, GN_Hmlpt)
#     b = np.einsum('mn, nije -> minje', kx ** 2, grad_GN_Hmlpt)
#     c = np.einsum('mnie, mnj -> minje', Hesskx, gkx)
#     d = np.einsum('mni, mnje -> minje', gkx, Hesskx)
#     return (-1 * a + b - c - d) / self.nParticles # multiplied by -1 because minus log posterior is used

# FOR TESTING PURPOSES REMOVE LATER
# def getSVN_deterministic_correction_numerical(self, gkx, HBD_inv_K, K_HBD_inv, dict=None):
#     # gradH_action_HBD_inv_K = self.gradHBD_action_mnbd(grad_hij_BD, HBD_inv_K)
#     X = dict['X'].flatten()
#     h = dict['h']
#     M = dict['M']
#     GN_Hmlpt = dict['GN_Hmlpt']
#     grad_numerical_b = nd.Gradient(self.form_h_ij_BD_for_numerical_derivative_extra_variables)(X, GN_Hmlpt, h, M)
#     grad_numerical_b_new = np.einsum('ijk -> kij', grad_numerical_b)
#     gradH_action_HBD_inv_K = np.einsum('abc, bc -> a', grad_numerical_b_new, self.reshapeNNDDtoNDND(HBD_inv_K)).reshape(self.nParticles, self.DoF)
#     a = self.gradK_action_mnbd(gkx, HBD_inv_K)
#     b = self.mnbd_mb_matvec(K_HBD_inv, gradH_action_HBD_inv_K) # bugs may be in this
#     return (a - b) * self.nParticles
#
# def makeFullMatrixFromBlockDiagonal(self, mbd):
#     res = np.zeros((self.dim, self.dim))
#     for m in range(self.nParticles):
#         res[m * self.DoF : self.DoF * (m + 1), m * self.DoF : self.DoF * (m + 1)] = mbd[m]
#     return res
# def form_h_ij_BD_for_numerical_derivative_extra_variables(self, X, GN_Hmlpt, bandwidth, metric):
#     # Input is an m*b vector
#     X = X.reshape(self.nParticles, self.DoF)
#     deltas = self.getDeltas(X, X)
#     metricDeltas = self.getMetricDeltas(metric, deltas)
#     deltaMetricDeltas = self.getDeltasMetricDeltas(deltas, metricDeltas)
#     kx = self.getKernelPiecewise(bandwidth, deltaMetricDeltas)
#     gkx = self.getGradKernelPiecewise(bandwidth, kx, metricDeltas)
#     h_ij_BD = self.h_ij_BD(GN_Hmlpt, kx, gkx)
#     H_BD = self.makeFullMatrixFromBlockDiagonal(h_ij_BD)
#     return H_BD
# def getUphillContribution(self, D_SVN,  gmlpt_new):
#     return self.mnbd_mb_matvec(D_SVN, -1 * gmlpt_new)

# def getDivergenceDiffusionContribution(self, gkx, gradHBD, HBD_inv_K, K_HBD_inv):
#     divK = np.mean(gkx, axis=0)
#     gradH_action_HBD_inv_K = self.gradHBD_mnbd(gradHBD, HBD_inv_K)
#     # a = self.gradK_action_mnbd(gkx, HBD_inv_K)
#     # a = self.gradK_action_mnbd_SUPERNAIVE(gkx, HBD_inv_K)
#     # a = self.gradK_mnbd(gkx, HBD_inv_K)
#     a = self.gradK_mnbd_new(gkx, HBD_inv_K)
#     b = self.mnbd_mb_matvec(K_HBD_inv, gradH_action_HBD_inv_K) # bugs may be in this
#     c = self.mnbd_mb_matvec(K_HBD_inv, divK)
#     return (a - b + c) * self.nParticles

# def getSVN_noiseContribution(self, kx, HBD_inv_sqrt):
#     np.random.seed(int(time()))
#     B = np.random.normal(0, 1, (self.nParticles, self.DoF))
#     tmp = self.mbd_mb_matvec(HBD_inv_sqrt, B)
#     return np.sqrt(2 * self.nParticles) * self.K_action_mb_vec(kx, tmp)

# def getSVN_noiseContribution_new(self, D_SVN):
#     np.random.seed(int(time()))
#     B = np.random.normal(0, 1, self.dim)
#     tmp = self.compute_cholesky_if_possible(2 * self.reshapeNNDDtoNDND(D_SVN))
#     return (tmp @ B).reshape(self.nParticles, self.DoF)

##################################################################
# Stochastic SVN methods: Exerimental / Lazy / Naive methods
##################################################################
# def gradK_action_mnbd_SUPERNAIVE(self, gkx, HBD_inv_K):
#     A = self.reshapeNNDDtoNDND(HBD_inv_K)
#     gK = self.gradK_single_parameter_def(gkx)
#     if self.iter_ == 0:
#         self.oecontract_gradK_SUPERNAIVE = oe.contract_expression('ijk, jk -> i', gK.shape, A.shape)
#     return self.oecontract_gradK_SUPERNAIVE(gK, A).reshape(self.nParticles, self.DoF)

# def gradK_single_parameter_def(self, gkx):
#     dim = self.nParticles * self.DoF
#     res = np.zeros((dim, dim, dim))
#     #############################################################
#     # Get the gradient of augmented kernel matrix
#     #############################################################
#     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
#         zeros = np.zeros((self.nParticles, self.DoF))
#         if m < n:
#             zeros[m] = copy.deepcopy(gkx[m, n])
#             zeros[n] = copy.deepcopy(-1 * gkx[m, n])
#         elif m > n:
#             zeros[n] = copy.deepcopy(-1 * gkx[m, n])
#             zeros[m] = copy.deepcopy(gkx[m, n])
#         zeros = zeros.flatten()
#         block = np.einsum('ij, z -> ijz', np.eye(self.DoF), zeros)
#         res[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1), :] = block
#     res /= self.nParticles
#     return res
# rep_a = np.mean(self.gradKernel, axis=0).flatten()
# rep_b = np.einsum('acc -> a', res)
# try:
#     assert np.allclose(rep_a, rep_b, rtol=self.rtol, atol=self.atol)
#     return res
# except:
#     return None


# def grad_hij_BD(self, Hmlpt, kx, gkx, hesskx):
#     if self.iter_ == 0:
#         self.gradH_bar_a = oe.contract_expression('xij, xz, xzk -> zijk', Hmlpt.shape, kx.shape, gkx.shape)
#         self.gradH_bar_b = oe.contract_expression('xzik, xzj -> zijk', hesskx.shape, gkx.shape)
#         # self.gradH_bar_c = oe.contract_expression('xzi, xzjk -> zijk', gkx.shape, hesskx.shape)
#     a = 2 * self.gradH_bar_a(-1 * Hmlpt, kx, gkx)
#     b = self.gradH_bar_b(hesskx, gkx)
#     # c = self.gradH_bar_c(gkx, hesskx)
#     c = np.einsum('xikj -> xkji', b) # We can reuse this calculation!
#     return (a - b - c) / self.nParticles

# def gradK_action_mnbd(self, gkx, Bnndd):
#     return np.einsum('nnbd, nmd -> mb', Bnndd, gkx) / self.nParticles
# Method to get gradK action
# def getGkxBlock(self, gkx):
#     gkx_block = np.zeros((self.nParticles, self.nParticles, self.dim))
#     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
#         zeros = np.zeros((self.nParticles, self.DoF))
#         zeros[m] = (gkx[m, n])
#         zeros[n] = (-1 * gkx[m, n])
#         # zeros[m] = copy.deepcopy(gkx[m, n])
#         # zeros[n] = copy.deepcopy(-1 * gkx[m, n])
#         zeros = zeros.flatten()
#         gkx_block[m, n, :] = zeros
#     return gkx_block
# def gradK_mnbd(self, gkx, mnbd):
#     mnbd = self.reshapeNNDDtoNDND(mnbd)
#     def c_to_f(v):
#         return (v.reshape(self.nParticles, self.DoF)).reshape(self.dim, order='F')
#     mat_c_to_f = lambda A: np.apply_along_axis(c_to_f, 0, A)
#     A_DDNN = self.reshapeNDNDtoDDNN(mat_c_to_f(mnbd))
#     gkx_block = self.getGkxBlock(gkx)
#     gradD_K_block = (gkx_block / self.nParticles).reshape(self.nParticles, self.nParticles, 1, self.DoF, self.nParticles).swapaxes(1,3).reshape(self.nParticles, 1, self.DoF, self.nParticles, self.nParticles)
#     return np.einsum('obdmn, qdmn -> oq', gradD_K_block, A_DDNN)

# def grad_hij_BD_new(self, Hmlpt, kx, gkx, hesskx): # (OUTDATED)
#     if self.iter_ == 0:
#         self.gradH_bar_a = oe.contract_expression('mn, mne, nij -> mije', kx.shape, gkx.shape, Hmlpt.shape)
#         self.gradH_bar_b = oe.contract_expression('mnie, mnj -> mije', hesskx.shape, gkx.shape)
#         self.gradH_bar_c = oe.contract_expression('mni, mnje -> mije', gkx.shape, hesskx.shape)
#     a = self.gradH_bar_a(2 * kx, gkx, Hmlpt)
#     b = self.gradH_bar_b(hesskx, gkx)
#     c = self.gradH_bar_c(gkx, hesskx)
#     return (a + b + c) / self.nParticles
# def gradHBD_action_mnbd(self, grad_hij_BD, Bnndd):
#     return np.einsum('nadb, nndb -> na', grad_hij_BD, Bnndd)
####################################################
# Calculations
####################################################
# H = self.H_bar(Hmlpt_new, kx, gkx)
# gradH = self.gradH(Hmlpt_new, kx, gkx, hesskx)
# # eigvals, eigvecs = self.findPosdefEigendecompisition(H)
# eigvals, eigvecs = np.linalg.eigh(H)
# # This order makes the correct inverse!!!
# H_inv = self.reshapeNDNDtoNNDD(eigvecs @ np.diag(1 / eigvals) @ eigvecs.T)
# H_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
# K_H_inv = self.getK_action_mat(kx, H_inv)
# D = self.getSVN_Diffusion(kx, K_H_inv)


#########################################################
# OLD SCRATCHWORK
#########################################################

# def test_gradH_form(self):
#     # Test if new indicies match old SVN-Hessian
#     test_a = (np.einsum('Nij, Nm, Nn -> mnij', self.GN_Hmlpt, self.kx, self.kx) + np.einsum('Nni, Nmj -> mnij', self.gkx, self.gkx)) / self.nParticles
#     assert np.allclose(test_a, self.stein.reshapeNDNDtoNNDD(self.H_bar), rtol=1e-6, atol=1e-6)
#
#     a = np.einsum('oije, om, on -> mnoije', self.grad_GN_Hmlpt, self.kx, self.kx) \
#         + np.einsum('oij, ome, on -> mnoije', self.GN_Hmlpt, self.gkx, self.kx) \
#         + np.einsum('oij, om, one -> mnoije', self.GN_Hmlpt, self.kx, self.gkx) \
#         + np.einsum('onie, omj -> mnoije', self.Hesskx, self.gkx) \
#         + np.einsum('oni, omje -> mnoije', self.gkx, self.Hesskx)
#
#     b = np.einsum('mo, Nij, Nme, Nn -> mnoije', self.delta_N, -1 * self.GN_Hmlpt, self.gkx, self.kx) \
#         - np.einsum('mo, Nni, Nmje -> mnoije', self.delta_N, self.gkx, self.Hesskx)
#
#     c = np.einsum('no, Nij, Nm, Nne -> mnoije', self.delta_N, -1 * self.GN_Hmlpt, self.kx, self.gkx) \
#         - np.einsum('no, Nnie, Nmj -> mnoije', self.delta_N, self.Hesskx, self.gkx)
#
#     res = (a + b + c) / self.nParticles
#     for m, n, o, i, j, e in itertools.product(range(self.nParticles), range(self.nParticles), range(self.nParticles), range(self.DoF), range(self.DoF), range(self.DoF)):
#         assert np.allclose(res[m,n,o,i,j,e], self.gradH[self.phi(m, i), self.phi(n, j), self.phi(o, e)], rtol=1e-6, atol=1e-6)

# def test_gradH_form_simpler(self):
#     # Testing equation for \nabla H rewritten with conventions
#     R = np.einsum('oije, om, on -> omnije', self.grad_GN_Hmlpt, self.kx, self.kx)
#     B = np.einsum('oij, ome, on -> onmije', self.GN_Hmlpt, self.gkx, self.kx)
#     C = np.einsum('onie, omj -> omnjie', self.Hesskx, self.gkx)
#
#     a = np.einsum('omnije -> mnoije', R + B) + np.einsum('onmije -> mnoije', B + C) + np.einsum('omnjie -> mnoije', C)
#     b = - np.einsum('mo, Nnmije -> mnoije', self.delta_N, B + C)
#     c = - (np.einsum('no, Nmnije -> mnoije', self.delta_N, B) + np.einsum('no, Nmnjie -> mnoije', self.delta_N, C))
#
#     test_a = (a + b + c) / self.nParticles
#     for m, n, o, i, j, e in itertools.product(range(self.nParticles), range(self.nParticles), range(self.nParticles), range(self.DoF), range(self.DoF), range(self.DoF)):
#         assert np.allclose(test_a[m,n,o,i,j,e], self.gradH[self.phi(m, i), self.phi(n, j), self.phi(o, e)], rtol=1e-6, atol=1e-6)

# def test_gradHBD_form_simpler(self):
#     # Test that $\nabla (H_{BD})^{mno}_{ije} = (\nabla H)^{mno}_{ije} \delta_{mn}$
#     R = np.einsum('oije, om, on -> omnije', self.grad_GN_Hmlpt, self.kx, self.kx)
#     B = np.einsum('oij, ome, on -> onmije', self.GN_Hmlpt, self.gkx, self.kx)
#     C = np.einsum('onie, omj -> omnjie', self.Hesskx, self.gkx)
#
#     a = np.einsum('omnije -> mnoije', R + B) + np.einsum('onmije -> mnoije', B + C) + np.einsum('omnjie -> mnoije', C)
#     b = - np.einsum('mo, Nnmije -> mnoije', self.delta_N, B + C)
#     c = - (np.einsum('no, Nmnije -> mnoije', self.delta_N, B) + np.einsum('no, Nmnjie -> mnoije', self.delta_N, C))
#
#     test_a = (a + b + c) / self.nParticles
#
#     test_a = np.einsum('mnoije, mn -> mnoije', test_a, self.delta_N)
#     for m, n, o, i, j, e in itertools.product(range(self.nParticles), range(self.nParticles), range(self.nParticles), range(self.DoF), range(self.DoF), range(self.DoF)):
#         assert np.allclose(test_a[m,n,o,i,j,e], self.gradHBD_numerical[self.phi(m, i), self.phi(n, j), self.phi(o, e)], rtol=1e-6, atol=1e-6)
#
#     # Test: Get bracket with \delta_{mn} explicit
#     bracket_test1 = self.nParticles * np.einsum('aA, ABe, Be -> a', self.A_BD_ndnd.T, self.gradHBD_numerical, self.A_BD_ndnd).reshape(self.nParticles, self.DoF)
#     bracket_test2 = self.nParticles * np.einsum('mnoije, noje, mpif -> pf', test_a, self.A_BD_nndd, self.A_BD_nndd)
#     np.allclose(bracket_test1, bracket_test2, rtol=1e-6, atol=1e-6)
#
#     # Test: Get bracket term with LHS action eliminating \delta_{mn}
#
#     R_block = np.einsum('oije, om -> omije', self.grad_GN_Hmlpt, self.kx ** 2)
#     B_block = np.einsum('oij, ome, om -> omije', self.GN_Hmlpt, self.gkx, self.kx)
#     C_block = np.einsum('omie, omj -> omjie', self.Hesskx, self.gkx)
#     # C_ije = np.einsum('omjie -> omije', C_block)
#
#     test_lhs_a = np.einsum('omije, moje -> mi', R_block + 2 * B_block + C_block, self.A_BD_nndd) \
#                 + np.einsum('omjie, moje -> mi', C_block, self.A_BD_nndd) \
#                 - np.einsum('omije, mmje -> mi', 2 * B_block + C_block, self.A_BD_nndd) \
#                 - np.einsum('omjie, mmje -> mi', C_block, self.A_BD_nndd)
#
#     test_lhs_b = self.nParticles * np.einsum('mnoije, noje -> mi', test_a, self.A_BD_nndd)
#     np.allclose(test_lhs_a, test_lhs_b, rtol=1e-6, atol=1e-6)



# test_a = GK - bracket
# test_b = GK_test - bracket_test
# Timing tests
# import functools
# import timeit
# t_contract = timeit.Timer(functools.partial(contract_ijemo, tmp2, self.A_BD_nndd, self.A_BD_nndd))
# t_contract = timeit.Timer(functools.partial(contract_ijemo, tmp2, self.A_BD_nndd, self.A_BD_nndd))

# test_b = self.nParticles * (np.einsum('abe, be -> a', self.gradK, self.A_BD_ndnd) -
#                             np.einsum('aA, ABe, Be -> a', self.A_BD_ndnd.T, self.gradHBD_numerical, self.A_BD_ndnd)).reshape(self.nParticles, self.DoF)

# assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)

# Another way to express test_b
# test_c = self.nParticles * (np.einsum('abe, bc, ce -> a', self.gradK, self.HBD_inv_ndnd, self.K) -
#                             np.einsum('ab, bA, ABe, Bc, ce -> a', self.K, self.HBD_inv_ndnd, self.gradHBD_numerical, self.HBD_inv_ndnd, self.K)).reshape(self.nParticles, self.DoF)


# def test_determinstic_correction_math(self):
#     R = np.einsum('oije, om, on -> omnije', self.grad_GN_Hmlpt, self.kx, self.kx)
#     B = np.einsum('oij, ome, on -> onmije', self.GN_Hmlpt, self.gkx, self.kx)
#     C = np.einsum('onie, omj -> omnjie', self.Hesskx, self.gkx)
#
#     tmp1 = B + C
#     tmp2 = tmp1 + R
#
#     bracket = np.einsum('onmije, mpif, noje -> pf', tmp2, self.A_nndd, self.A_nndd) \
#             + np.einsum('omnije, mpif, noje -> pf', B, self.A_nndd, self.A_nndd) \
#             + np.einsum('omnjie, mpif, noje -> pf', C, self.A_nndd, self.A_nndd) \
#             - np.einsum('onmije, mpif, nmje -> pf', tmp1, self.A_nndd, self.A_nndd) \
#             - np.einsum('omnije, mpif, nnje -> pf', B, self.A_nndd, self.A_nndd) \
#             - np.einsum('omnjie, mpif, nnje -> pf', C, self.A_nndd, self.A_nndd) \
#             + np.einsum('mne, nmie -> mi', self.gkx, self.A_nndd) \
#             - np.einsum('mne, nnie -> mi', self.gkx, self.A_nndd)
#
#     test_b = self.stein.getDeterministicCorrection_simplemath(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_nndd)
#     assert np.allclose(test_b, bracket, rtol=1e-6, atol=1e-6)

# def test_deterministic_correction(self):
#     # Testing \nabla K : A
#     test_a = np.einsum('mne, nmie -> mi', self.gkx, self.A_nndd) - np.einsum('mne, nnie -> mi', self.gkx, self.A_nndd)
#     test_b = self.nParticles * np.einsum('abc, bc -> a', self.gradK, self.A_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)




# def test_gradK_numerically(self):
#     # Test that gradK is numerically correct, and that it produces the correct divergence term.
#     X_flattened = self.X.flatten()
#     grad_numerical = nd.Gradient(self.formK_for_numerical_derivative)(X_flattened)
#     grad_numerical = np.einsum('ijk -> kij', grad_numerical) # Recall that numdifftools puts derivative index first
#     assert np.allclose(self.gradK, grad_numerical, rtol=1e-6, atol=1e-6)
#     rep_a = np.mean(self.gkx, axis=0).flatten()
#     rep_b = np.einsum('acc -> a', grad_numerical)
#     assert np.allclose(rep_a, rep_b, rtol=1e-6, atol=1e-6)

# def test_deterministic_correction_a(self):
#     test_a = np.einsum('mne, nmie -> mi', self.gkx, self.A_nndd) \
#             - np.einsum('mne, nnie -> mi', self.gkx, self.A_nndd)
#     test_b = self.nParticles * np.einsum('abc, bc -> a', self.gradK, self.A_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)
#     test_c = self.stein.getDeterministicCorrection_a(self.gkx, self.A_nndd)
#     assert np.allclose(test_a, test_c, rtol=1e-6, atol=1e-6)

# def test_deterministic_correction_b(self):
#     # this guy is kept the same!
#     a_tmp = np.einsum('oije, om, on -> mnoije', self.grad_GN_Hmlpt, self.kx, self.kx) \
#         + np.einsum('oij, ome, on -> mnoije', self.GN_Hmlpt, self.gkx, self.kx) \
#         + np.einsum('oij, om, one -> mnoije', self.GN_Hmlpt, self.kx, self.gkx) \
#         + np.einsum('onie, omj -> mnoije', self.Hesskx, self.gkx) \
#         + np.einsum('oni, omje -> mnoije', self.gkx, self.Hesskx)
#
#     a = np.einsum('mnoije, mpif, noje -> pf', a_tmp, self.A_nndd, self.A_nndd)
#
#     # This guy requires a bit of modification
#     b_tmp = np.einsum('Nij, Noe, Nn -> noije', -1 * self.GN_Hmlpt, self.gkx, self.kx) \
#         - np.einsum('Nni, Noje -> noije', self.gkx, self.Hesskx)
#     b = np.einsum('noije, opif, noje -> pf', b_tmp, self.A_nndd, self.A_nndd)
#
#     c_tmp = np.einsum('Nij, Nm, Nne -> mnije', -1 * self.GN_Hmlpt, self.kx, self.gkx) \
#         - np.einsum('Nnie, Nmj -> mnije', self.Hesskx, self.gkx)
#     c = np.einsum('mnije, mpif, nnje -> pf', c_tmp, self.A_nndd, self.A_nndd)
#
#     test_a = a + b + c
#
#     test_b = self.nParticles * np.einsum('ab, bce, ce -> a', self.A_ndnd.T, self.gradH_numerical, self.A_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)
#     test_c = self.stein.getDeterministicCorrection_b(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_nndd)
#     assert np.allclose(test_a, test_c, rtol=1e-6, atol=1e-6)
#########################################################################
# Stochastic SVN: Action methods (5)
#########################################################################

##################################################################
# Stochastic SVN Block Diagonal: Matrix-vector product methods (2)
##################################################################

# def test_deterministic_correction_numerically(self):
#     # Check if deterministic correction is correct by confirming the following:
#     # D^{SVN} \nabla \ln \pi + \nabla \cdot D^{SVN} = v^{SVN} + v^{cor}
#     X_flattened = self.X.flatten()
#     grad_numerical = nd.Gradient(self.form_D_SVN_for_numerical_derivative)(X_flattened)
#     grad_numerical = np.einsum('ijk -> kij', grad_numerical)
#     div_D_SVN = np.einsum('abb -> a', grad_numerical).reshape(self.nParticles, self.DoF)
#     uphill = (self.D_SVN @ (-1 * self.gmlpt).flatten()).reshape(self.nParticles, self.DoF)
#     a = uphill + div_D_SVN
#     v_svn = self.stein.getSVN_Direction(self.kx, self.gkx, self.gmlpt, self.K_HBD_inv, self.HBD_inv_K)
#     v_deterministic_correction = self.stein.getSVN_deterministic_correction(self.gkx, self.grad_hij_BD, self.HBD_inv_K, self.K_HBD_inv)
#     b = v_svn + v_deterministic_correction
#     assert(np.allclose(a, b, rtol=1e-8, atol=1e-8))


# C_ije = np.einsum('omjie -> omije', C)

# assert np.allclose(R, R_test2, rtol=1e-6, atol=1e-6)
# assert np.allclose(B, B_test2, rtol=1e-6, atol=1e-6)
# assert np.allclose(C, C_test2, rtol=1e-6, atol=1e-6)
# Test: H^{-1} K
# test1 = self.HBD_inv_ndnd @ self.K
# assert np.allclose(self.A_BD_ndnd, test1, rtol=1e-6, atol=1e-6)

# Test: Block values of R, B, C
# R_test1 = np.einsum('oije, om, on -> omnije', self.grad_GN_Hmlpt, self.kx, self.kx)
# B_test1 = np.einsum('oij, ome, on -> onmije', self.GN_Hmlpt, self.gkx, self.kx)
# C_test1 = np.einsum('onie, omj -> omnjie', self.Hesskx, self.gkx)

# R_test2 = R_test1[:, range(self.nParticles), range(self.nParticles)]
# B_test2 = B_test1[:, range(self.nParticles), range(self.nParticles)]
# C_test2 = C_test1[:, range(self.nParticles), range(self.nParticles)]

# Calculate temp variables
# bracket = np.einsum('omije, mpif, moje -> pf', tmp2, self.A_BD_nndd, self.A_BD_nndd) \
#         - np.einsum('omije, mpif, mmje -> pf', tmp1, self.A_BD_nndd, self.A_BD_nndd) \
#         + np.einsum('omjie, mpif, moje -> pf', C, self.A_BD_nndd, self.A_BD_nndd) \
#         - np.einsum('omjie, mpif, mmje -> pf', C, self.A_BD_nndd, self.A_BD_nndd)

# assert np.allclose(bracket, bracket_check, rtol=1e-6, atol=1e-6)
# def test_getBD_Deterministic_correction_dirtycode(self):
#     R = np.einsum('oije, om -> omije', self.grad_GN_Hmlpt, self.kx ** 2)
#     B = np.einsum('oij, ome, om -> omije', self.GN_Hmlpt, self.gkx, self.kx)
#     C = np.einsum('omie, omj -> omjie', self.Hesskx, self.gkx)
#
#     tmp1 = 2 * B + C
#     tmp2 = R + copy.deepcopy(tmp1)
#
#     contract_ijemo = oe.contract_expression('omije, mpif, moje -> pf', tmp2.shape, self.A_BD_nndd.shape, self.A_BD_nndd.shape)
#     contract_ijemm = oe.contract_expression('omije, mpif, mmje -> pf', tmp1.shape, self.A_BD_nndd.shape, self.A_BD_nndd.shape)
#     contract_jiemo = oe.contract_expression('omjie, mpif, moje -> pf', C.shape, self.A_BD_nndd.shape, self.A_BD_nndd.shape)
#     contract_jiemm = oe.contract_expression('omjie, mpif, mmje -> pf', C.shape, self.A_BD_nndd.shape, self.A_BD_nndd.shape)
#
#     bracket_check = contract_ijemo(tmp2, self.A_BD_nndd, self.A_BD_nndd) \
#                     - contract_ijemm(tmp1, self.A_BD_nndd, self.A_BD_nndd) \
#                     + contract_jiemo(C, self.A_BD_nndd, self.A_BD_nndd) \
#                     - contract_jiemm(C, self.A_BD_nndd, self.A_BD_nndd)
#
#     # Test: Check that first contribution to deterministic correction is good
#     GK = np.einsum('mne, nmie -> mi', self.gkx, self.A_BD_nndd) - np.einsum('mne, nnie -> mi', self.gkx, self.A_BD_nndd)
#     GK_test = self.nParticles * np.einsum('abe, be -> a', self.gradK, self.A_BD_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(GK, GK_test, rtol=1e-6, atol=1e-6)
#
#     # Test: Calculate and compare different direct calculations for deterministic correction
#     bracket_test1 = self.nParticles * np.einsum('aA, ABe, Be -> a', self.A_BD_ndnd.T, self.gradHBD_numerical, self.A_BD_ndnd).reshape(self.nParticles, self.DoF)
#     bracket_test2 = self.nParticles * np.einsum('aA, ABe, Be -> a', self.A_BD_ndnd.T, self.gradHBD, self.A_BD_ndnd).reshape(self.nParticles, self.DoF)
#     assert np.allclose(bracket_test1, bracket_test2, rtol=1e-6, atol=1e-6)
#
#     # Test: Check that the method calculates term (ii) correctly
#     assert np.allclose(bracket_check, bracket_test1, rtol=1e-6, atol=1e-6)
#
#     det_cor = GK - bracket_check
#     assert np.allclose(det_cor, GK_test - bracket_test1, rtol=1e-6, atol=1e-6)
#
#     test_direct = self.stein.getBD_Deterministic_correction(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_BD_nndd)
#     assert np.allclose(test_direct, GK_test - bracket_test1, rtol=1e-6, atol=1e-6)
# def test_form_gradHBD(self):
#     R = np.einsum('oije, om -> omije', self.grad_GN_Hmlpt, self.kx ** 2)
#     B = np.einsum('oij, ome, om -> omije', self.GN_Hmlpt, self.gkx, self.kx)
#     C = np.einsum('omie, omj -> omjie', self.Hesskx, self.gkx)
#     C_ije = np.einsum('omjie -> omije', C)
#     tmp1 = R + 2 * B
#     tmp2 = 2 * B + C
#     a = np.einsum('No, Nmije -> moije', self.delta_N, tmp1)
#     b = np.einsum('mo, Nmije -> moije', self.delta_N, tmp2)
#     c = np.einsum('mo, Nmjie -> moije', self.delta_N, C)
#     test_a = (a + b + c) / self.nParticles
#     for m, o, i, j, e in itertools.product(range(self.nParticles), range(self.nParticles), range(self.DoF), range(self.DoF), range(self.DoF)):
#         assert np.allclose(test_a[m,o,i,j,e], self.gradH_numerical[self.phi(m, i),  self.phi(o, e)], rtol=1e-6, atol=1e-6)

# test1 = hr.test_x_graph_index_python()
# n1 = 2
# n2 = 1
# mu_rosen = 1
# a = 1 / 20
# b = np.ones((n2, n1-1)) * (100 / 20)
# hr = hybrid_rosenbrock(n1, n2, mu_rosen, a, b)
# hr.newDrawFromLikelihood(10)
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# tex_fonts = {
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "serif",
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": 10,
#     "font.size": 10,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8
# }
# plt.rcParams.update(tex_fonts)
# width = 469.75502
# plt.rcParams['figure.figsize']=set_size(width, fraction=1/2, subplots=(1, 1))
# plt.arrow(5, -0.003, 0.1, 0, width=0.015, color="k", clip_on=False, head_width=0.12, head_length=0.12)
# plt.arrow(0.003, 5, 0, 0.1, width=0.015, color="k", clip_on=False, head_width=0.12, head_length=0.12)

# sns.set_palette(sns.color_palette(colors))
# array = np.array([1., 2.])
# array = np.array([1.1, 2.2])
# res1 = hr.getMinusLogLikelihood_individual(array)
# fig, ax = plt.subplots()
# particles = hr.drawSample(100000)
# particles_dataframe = pd.DataFrame(particles)
# obj = sns.pairplot(particles_dataframe, diag_kind='kde', plot_kws={"s": 3})
# obj.fig.show()
# import corner

# fig = corner.corner(particles)

# fig.show()
# pass
# res1_test = hr.getMinusLogLikelihood_individual_new(array)
# delta = hr.formKroneckerMatricies()
# res2 = hr.getMinusLogLikelihood_individual(array)
# grad1 = hr.getGradientMinusLogLikelihood_individual(array)
# grad2 = hr.getGradientMinusLogLikelihood_individual_new(array)

# hessian1 = hr.getHessianMinusLogLikelihood_individual(array)
# hessian2 = hr.getHessianMinusLogLikelihood_individual_algo(array)

# Hessians agree. make sure code removes terms from working code
# gradientGNhessian1 = hr.getGradientGNHessianMinusLogLikelihood_individual(array)
# gradientGNhessian2 = hr.getGradientGNHessianMinusLogLikelihood_individual_algo(array)
# a = 1 + 1
#  res2 = hr.getMinusLogLikelihood_individual_naive(array)
# dum = 1 + 1

# def getGNHessianMinusLogLikelihood_individual(self, theta):
#     J = jacobian(self.getMinusLogLikelihood_individual)(theta)
#     self.nFisherLikelihoodEvaluations += 1
#     return np.outer(J, J)

# def getMinusLogLikelihood_individual(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     # b = np.ones((self.n2, self.n1-1))
#     term_a = self.a * (x_in[0] - self.mu_rosen) ** 2
#     term_b = np.sum(self.b * (x_graph[:, 1:] - x_graph[:, :-1] ** 2) ** 2)
#     return term_a + term_b
# def get_residuals(self, x_in):
#     x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#     r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#     rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#     gr1 = np.zeros(self.DoF)
#     gr1[0] = np.sqrt(2 * self.a)
#     grji =
#     res_dict = {'r1': r1, 'rji': rji}
#     return res_dict

# def getMinusLogLikelihood_individual(self, x_in, res_dict=None):
#     if res_dict is None:
#         x_graph = np.insert(x_in[1:].reshape(self.n2, self.n1-1), 0, x_in[0], axis=1)
#         r1 = np.sqrt(2 * self.a) * (x_in[0] - self.mu_rosen)
#         rji = np.sqrt(2 * self.b) * (x_graph[:, 1:] - x_graph[:, :-1] ** 2)
#         # return -np.log(self.z_inv) + r1 ** 2 / 2 + np.sum(rji ** 2) / 2
#         return r1 ** 2 / 2 + np.sum(rji ** 2) / 2
#     else:
#         r1 = res_dict['r1']
#         rji = res_dict['rji']
#         return r1 ** 2 / 2 + np.sum(rji ** 2) / 2

# def mu_python(self, j, i):
#     j += 1
#     i += 1
#     if i == 1:
#         return 1
#     elif i > 1:
#         return (j - 1) * (self.n1 - 1) + i

# def test_x_graph_index_python(self):
#     x = np.zeros((self.n2, self.n1))
#     for j, i in itertools.product(range(self.n2), range(self.n1)):
#         x[j,i] = self.mu_python_test1(j,i)
#     return x

# def mu_test_python(self, p, q):
#     if q == 0:
#         return 0
#     elif q > 0:
#         return self.n2 * p + q

# def mu_test_math(self, p, q):
#     if q == 1:
#         return 1
#     else:
#         return self.n2 * (p-1) + q
# def test_x_graph_index_python(self):
#     x = np.zeros((self.n2, self.n1))
#     for p, q in itertools.product(range(self.n2), range(self.n1)):
#         x[p,q] = self.mu_test_python(p,q)
#     return x

# def test_x_graph_index_math(self):
#     x = np.zeros((self.n2, self.n1))
#     range1 = np.arange(1, self.n2+1)
#     range2 = np.arange(1, self.n1+1)
#     for p, q in itertools.product(range1, range2):
#         x[p-1,q-1] = self.mu_test_math(p,q)
#     return x
# rangep = np.arange(0, self.n2 + 1, 1)
# rangeq = np.arange(2, self.n1 + 1, 1)

# # Algorithmic differentiation
# def getGradientMinusLogLikelihood_individual_algo(self, theta):
#     # ALGORITHMIC DIFFERENTIATION
#     theta = theta.astype('float64')
#     self.nGradLikelihoodEvaluations += 1
#     return jacobian(self.getMinusLogLikelihood_individual)(theta)
#
# def getHessianMinusLogLikelihood_individual_algo(self, theta):
#     # ALGORITHMIC DIFFERENTIATION
#     theta = theta.astype('float64')
#     self.nHessLikelihoodEvaluations += 1
#     return hessian(self.getMinusLogLikelihood_individual)(theta)
#


# def formKroneckerMatricies(self):
#     tmp1 = np.arange(1, self.n2 * (self.n1-1) + 1).reshape(self.n2, self.n1-1)
#     tmp2 = np.insert(tmp1, 0, 0, axis=1)
#     delta = np.zeros((self.n2, self.n1, self.DoF))
#     for j, i in itertools.product(range(self.n2), range(self.n1)):
#         delta[j, i, tmp2[j, i]] = 1
#     return_dict = {'delta1': delta[:, 1:], 'delta2': delta[:, :-1]}
#     return return_dict

# def mu(self, j, i):
#     return self.x_graph_index[j, i]
# def getxGraph_index(self):
#     tmp1 = np.arange(1, self.n2 * (self.n1-1) + 1).reshape(self.n2, self.n1-1)
#     return np.insert(tmp1, 0, 0, axis=1)

# def getGradientGNHessianMinusLogLikelihood_individual_algo(self, theta):
#     # ALGORITHMIC DIFFERENTIATION
#     theta = theta.astype('float64')
#     self.nHessLikelihoodEvaluations += 1
#     # getGradientGNHessianMinusLogLikelihood_individual(self, x_in):
#     return np.einsum('ijk -> kij', nd.Gradient(self.getGNHessianMinusLogLikelihood_individual)(theta))
# #     # return np.einsum('ijk -> kij', jacobian(self.getGNHessianMinusLogLikelihood_individual)(theta))


# g._legend.set_title('') # Remove 'dataset' Title in legend
# handles = g._legend_data.values()
# labels = g._legend_data.keys()
# g._legend.remove()
# g.fig.legend(handles=handles, labels=labels, frameon=False, bbox_to_anchor=(0.75, 0.75))
# g.fig.legend(handles=handles, labels=labels, ncol=len(labels), frameon=False, bbox_to_anchor=(0.75, 0.75))
# new_labels = ['sSVN', 'Truth']

# new_labels = ['sSVN', 'sSVN', 'Truth']
# for t, l in zip(obj._legend.texts, new_labels): t.set_text(l)
# obj = sns.pairplot(samples_GT_dataframe, plot_kws={"s": 3}, corner=True, diag_kind='kde')
# plt.rcParams["legend.borderpad"]=1
# var_names.append('x_%i' % d)

# replacements = {'sepal_length': r'$\alpha$', 'sepal_width': 'sepal',
#                 'petal_length': r'$\beta$', 'petal_width': 'petal',
#                 'versicolor': 'bloop'}

# concatenated =
# samples_dataframe = pd.DataFrame(samples)
# # samples_GT_dataframe_2 = pd.DataFrame(samples_GT[0:num] + 0.2)
# concatenated = pd.concat([GT_dataframe.assign(dataset='Truth'),
#                               samples_dataframe.assign(dataset='sSVGD')])
# concatenated = pd.concat([particles_svgd_dataframe.assign(dataset='sSVGD'),
#                           particles_svn_dataframe.assign(dataset='sSVN'),
#                               samples_GT_dataframe.assign(dataset='Truth')])

# else:
#     with h5py.File(file_read, 'r') as f:
#         iter_window_max = f['metadata']['total_num_iterations'][()]
#         iter_window_min = int(np.floor(iter_window_max * .75))
#         n = f['metadata']['nParticles'][()]
#         d = f['metadata']['DoF'][()]
#
#         window = int(iter_window_max - iter_window_min)
#         particles = np.zeros((window * n, d))
#         # arrangement = np.arange()
#         for l in range(window):
#             try:
#                 particles[l * n : n * (l + 1), 0 : d] = f['%i' % (iter_window_max - 1 - l)]['X'][()]
#             except:
#                 pass
#     samples_GT_dataframe = pd.DataFrame(samples_GT)
#     particles_dataframe = pd.DataFrame(particles)
#     concatenated = pd.concat([particles_dataframe.assign(dataset='Output'),
#                               samples_GT_dataframe.assign(dataset='Truth')])
#     obj = sns.pairplot(concatenated, plot_kws={"s": 3}, hue='dataset', corner=True, diag_kind='kde')
#     obj._legend.set_title('')
#     new_labels = ['Truth', 'sSVGD']
#     for t, l in zip(obj._legend.texts, new_labels): t.set_text(l)
#     obj.fig.savefig(file_save)
# df12 = particles_svgd_dataframe.append(particles_svn_dataframe)
# df12 = samples_GT_dataframe.append(particles_dataframe)
# sns.pairplot(particles_svn_dataframe, diag_kind='kde', corner=True, plot_kws={"s": 5, 'markers':'+'})
# sns.pairplot(particles_svgd_dataframe, diag_kind='kde', corner=True, plot_kws={"s": 5, 'markers':'+'})
# obj = sns.pairplot(samples_GT_dataframe, diag_kind='kde', corner=True, plot_kws={"s": 5, 'markers':'+'})
# obj.fig.show()
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# tex_fonts = {
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "serif",
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": 10,
#     "font.size": 10,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8
# }
# plt.rcParams.update(tex_fonts)

###############################
# Get output samples
###############################
# if outdir is None:
# with h5py.File(file_svgd, 'r') as f:
# with h5py.File(file_svgd, 'r') as f:
#     iter_window_max = f['metadata']['total_num_iterations'][()]
#     iter_window_min = int(np.floor(iter_window_max * .75))
#     n = f['metadata']['nParticles'][()]
#     d = f['metadata']['DoF'][()]
#
#     window = int(iter_window_max - iter_window_min)
#     particles_svgd = np.zeros((window * n, d))
#     # arrangement = np.arange()
#     for l in range(window):
#         try:
#             particles_svgd[l * n : n * (l + 1), 0 : d] = f['%i' % (iter_window_max - 1 - l)]['X'][()]
#         except:
#             pass
#
# with h5py.File(file_svn, 'r') as f:
#     iter_window_max = f['metadata']['total_num_iterations'][()]
#     iter_window_min = int(np.floor(iter_window_max * .75))
#     n = f['metadata']['nParticles'][()]
#     d = f['metadata']['DoF'][()]
#
#     window = int(iter_window_max - iter_window_min)
#     particles_svn = np.zeros((window * n, d))
#     # arrangement = np.arange()
#     for l in range(window):
#         try:
#             particles_svn[l * n : n * (l + 1), 0 : d] = f['%i' % (iter_window_max - 1 - l)]['X'][()]
#         except:
#             pass


# With all the distributions on top of each other
# num = np.min([particles_svn.shape[0], samples_GT.shape[0], particles_svgd.shape[0]])
# particles_svn_dataframe = pd.DataFrame(particles_svn[0:num])
# particles_svgd_dataframe = pd.DataFrame(particles_svgd[0:num])
# num = 1000
# n1 = 2
# n2 = 1
# mu_rosen = 1
# a = 1 / 20
# b = np.ones((n2, n1-1)) * (100 / 20)
# HRD = hybrid_rosenbrock(n1, n2, mu_rosen, a, b)
# from time import time
# np.random.seed(int(time()))
# # particles = HRD.newDrawFromLikelihood(2000000)



# def newDrawFromLikelihood(self, N):
#     samples = np.zeros((N, self.DoF))
#     for k in range(N):
#         samples[k, 0] = np.random.normal(self.mu_rosen, 1 / (2 * self.a))
#         # for j, I in itertools.product(range(self.n2), np.arange(1, self.n1)):
#         #         samples[k, self.mu(j, I)] = np.random.normal(samples[k, self.mu(j, I - 1)] ** 2, 1 / (2 * self.b[j,I-1]))
#         #         pass
#         for m, n in itertools.product(range(self.n2), range(self.n1-1)):
#             samples[k, self.mu(j, I)] = np.random.normal(samples[k, self.mu(j, I - 1)] ** 2, 1 / (2 * self.b[j,I-1]))
#             pass
#     return samples



# if self.kernel_metric == 'Fisher':
#     metric_new = np.mean(GNHmlpt_new, axis=0)
# else:
#     metric_new = np.eye(self.DoF)
# deltas = self.getDeltas(X_new, X_new)
# metricDeltas = self.getMetricDeltas(metric_new, deltas)
# deltaMetricDeltas = self.getDeltasMetricDeltas(deltas, metricDeltas)
#################################################
# Bandwidth selection uses previous bandwidth as starting point in search.
# Therefore for first iteration we must set a starting point at iteration 0.
#################################################
# if iter_ == 0:
#     h_med = self.getBandwidth(deltaMetricDeltas)
#     h_prev = copy.deepcopy(h_med)
# else:
#     h_med = self.getBandwidth(deltaMetricDeltas)
#     h_prev = copy.deepcopy(h)
#################################################
# Bandwidth selection.
#################################################
# if self.bw_selection == 'med':
#     # h = copy.deepcopy(h_med)
#     h = self.DoF * 2
#     kx = self.getKernelPiecewise(h, deltaMetricDeltas)
#     gkx = self.getGradKernelPiecewise(h, kx, metricDeltas)
#     hesskx = None
# else:
#     input_dict = {'h_prev': copy.deepcopy(h_prev), 'metric': copy.deepcopy(metric_new),
#                   'metricDeltas':copy.deepcopy(metricDeltas), 'deltaMetricDeltas': copy.deepcopy(deltaMetricDeltas),
#                   'X':copy.deepcopy(X_new)}#, 'gmlpt': copy.deepcopy(gmlpt_new)}
#     if self.bw_selection == 'BM':
#         # Brownian motion method of Wang 2020
#         # BW_cost_function = self.BM_cost
#         BW_cost_function = MMD
#     elif self.bw_selection == 'HE':
#         # Heat equation method from Liu 2020
#         BW_cost_function = self.HE_cost
#     elif self.bw_selection == 'NOISY_SVGD':
#         # Noisy BW selection
#         BW_cost_function = self.BM_cost_NOISY_SVGD
#     tpqo_dict = self.three_point_quadratric_optimize_NEW(copy.deepcopy(h_prev), BW_cost_function, input_dict)
#     h = tpqo_dict['h']
#     kx = tpqo_dict['kx']
#     gkx = tpqo_dict['gkx']
#     hesskx = tpqo_dict['hesskx']
# h_med = self.bandwidth_MED(X_new)
# h = self.getBandwidth_new(X_new, h, M=metric_new)
# kx, gkx, hesskx = getKernelWithDerivatives(X_new, h, metric_new, get_hesskx=True)
####################################################
# Evaluating SVGD direction / solving system for SVN direction.
####################################################
# mgJ = self.mgJ_new(kx, gkx, gmlpt_new)
# mgJ = self.mgJ_new_temp(kx, gkx, gmlpt_new, copy.copy(self.iter_), copy.copy(self.nIterations))
# if self.optimizeMethod == 'SVN':
#     H_bar = self.H_bar(GNHmlpt_new, kx, gkx)
#     alphas = self.solveSystem_new(H_bar, mgJ)
#     wx = self.w(alphas, kx)
#################################################
# Stepsize selection.
#################################################
# if self.stepsize_selection == 'constant':
#     eps = copy.deepcopy(self.step_hyperparameter)
# elif self.stepsize_selection == 'linesearch':
#     if self.optimizeMethod == 'SVGD':
#         gwx = self.getJacobianMapSVGD(h, kx, gkx, metric_new, metricDeltas, gmlpt_new, hesskx)
#         # TESTING NEW LINESEARCH METHOD
#         eps = self.linesearch_armijo(X_new, mgJ, gwx, gmlpt_new)
#         # eps = self.linesearch_new(X_new, mgJ, gwx, copy.copy(self.step_hyperparameter))
#     elif self.optimizeMethod == 'SVN':
#         gwx = self.grad_w_new(alphas, gkx)
#         eps = self.linesearch_armijo(X_new, wx, gwx, gmlpt_new)
# eps = self.linesearch_new(X_new, wx, gwx, copy.copy(self.step_hyperparameter))

#################################################
# For debugging purposes
#################################################
# self.DEBUG_dict[iter_] = self.debug_dict_init()
# self.DEBUG_dict[iter_]['kx'] = copy.deepcopy(kx)
# self.DEBUG_dict[iter_]['gkx'] = copy.deepcopy(gkx)
# self.DEBUG_dict[iter_]['deltas'] = copy.deepcopy(deltas)
# self.DEBUG_dict[iter_]['metricDeltas'] = copy.deepcopy(metricDeltas)
# self.DEBUG_dict[iter_]['deltasMetricDeltas'] = copy.deepcopy(deltaMetricDeltas)
# self.DEBUG_dict[iter_]['mgJ'] = copy.deepcopy(mgJ)
# self.DEBUG_dict[iter_]['h'] = copy.deepcopy(h)
# self.DEBUG_dict[iter_]['h_med'] = copy.deepcopy(h_med)
# self.DEBUG_dict[iter_]['X'] = copy.deepcopy(X_new)
# self.DEBUG_dict[iter_]['gmlpt'] = copy.deepcopy(gmlpt_new)

# if self.optimizeMethod == 'SVN':
#     X_new += eps * wx
# elif self.optimizeMethod == 'SVGD':
#     X_new += eps * mgJ
#     if self.SVGD_stochastic_correction is True:
#         # noise_correction_new
#         # X_new += self.getSVGD_noise_correction_NEW(kx, eps)['noise_correction'] # WORKS
#         X_new += self.getSVGD_noise_correction_MANUAL(kx, eps) # FOR TESTING PURPOSES

# Convergence criteria!
# if self.stepsize_selection == 'linesearch':
#     if self.isConverged() == True:
#         pass
# break
#################################################
# For debugging purposes
#################################################
# self.DEBUG_final_kx = copy.deepcopy(kx)
# self.DEBUG_final_gkx = copy.deepcopy(gkx)
# self.DEBUG_final_deltas = copy.deepcopy(deltas)
# self.DEBUG_final_metricDeltas = copy.deepcopy(metricDeltas)
# self.DEBUG_final_deltasMetricDeltas = copy.deepcopy(deltaMetricDeltas)
# self.DEBUG_final_bandwidth = copy.deepcopy(h)
# self.DEBUG_final_mgJ = copy.deepcopy(mgJ)
# self.DEBUG_dict[iter_ + 1] = {'X_final': copy.deepcopy(X_new)}
# dd.io.save(os.path.join(self.RUN_OUTPUT_DIR, '%i_iterations.h5' % iter_), self.DEBUG_dict)
# dd.io.save(os.path.join(self.RUN_OUTPUT_DIR, '%i_iterations_tpqo.h5' % iter_), self.DEBUG_tpqo)
# g.create_dataset('bandwidthselection', data=copy.deepcopy(self.bw_selection))
# g.create_dataset('stepsizeselection', data=copy.deepcopy(self.stepsize_selection))
# g.create_dataset('maxstep', data=copy.deepcopy(self.step_hyperparameter))
# g.create_dataset('opt', data=copy.deepcopy(self.optimizeMethod))
# g.create_dataset('nLikelihoodEvals', data=copy.deepcopy(self.model.nLikelihoodEvaluations))
# g.create_dataset('nGradLikelihoodEvals', data=copy.deepcopy(self.model.nGradLikelihoodEvaluations))
# g.create_dataset('nFisherLikelihoodEvals', data=copy.deepcopy(self.model.nFisherLikelihoodEvaluations))


# h =
# h = 10000
# h = self.bandwidth_MED(X)
# h = 0.01
# self.DEBUG_dict = {}
# self.DEBUG_tpqo = {}


# assert(self.optimizeMethod == 'SVGD' or self.optimizeMethod == 'SVN')
# assert(self.bw_selection == 'HE' or self.bw_selection == 'BM' or self.bw_selection == 'med'
#        or self.bw_selection == 'NOISY_SVGD')
# assert(self.stepsize_selection == 'linesearch' or self.stepsize_selection == 'constant')
# assert(self.kernel_metric == 'Id' or self.kernel_metric == 'Fisher')
# assert(self.step_hyperparameter > 0)
# assert(self.nParticles > 0)
# assert(self.DoF > 0)
# assert(self.nIterations > 0)
# if self.SVGD_stochastic_correction is True:
#     if self.stepsize_selection == 'linesearch':
#         raise Exception('ERROR: Linesearch not implemented for stochastic SVGD')
#     if self.optimizeMethod == 'SVN':
#         raise Exception('ERROR: Stochastic SVGD enabled with SVN optimizer')
# self.SVGD_stochastic_correction = True
# self.SVGD_stochastic_correction = False
# self.bw_selection = 'NOISY_SVGD'
# self.bw_selection = 'HE'

# Stochastic settings
# self.bw_selection = 'med'
# self.stepsize_selection = 'constant'
# self.step_hyperparameter = 0.01
# self.kernel_metric = 'Id'

# Deterministic settings
# self.bw_selection = 'BM'
# self.bw_selection = 'HE'
# self.bw_selection = 'med'
# self.stepsize_selection = 'linesearch'
# self.step_hyperparameter = 0.5 # SVN
# self.step_hyperparameter = 0.1 # SVGD

# self.kernel_metric = 'Id'

# self.kernel_metric = 'Fisher'
# self.step_hyperparameter = 1
# self.step_hyperparameter = 0.1
# self.step_hyperparameter = 0.01
# self.step_hyperparameter = 0.0025


# Debugging stuff removed

def debug_dict_init(self):
    dict = {}
    dict['kx'] = None
    dict['gkx'] = None
    dict['h'] = None
    dict['mgJ'] = None
    dict['deltas'] = None
    dict['metricDeltas'] = None
    dict['deltasMetricDeltas'] = None
    dict['X'] = None
    dict['gmlpt'] = None
    return dict

def debug_tpqo_dict_init(self):
    dict = {}
    dict['cost_dict0'] = None
    dict['cost_dict1'] = None
    dict['cost_dict2'] = None
    dict['gCost0'] = None
    dict['cost0eps'] = None
    dict['s'] = None
    return dict



    def isConverged(self):
        if self.iter_ == 0:
            self.isConvergedCounter = 0
        n = 10
        max_halfing = 5
        tol = 0.01
        if self.iter_ > n:
            average0 = 0
            average1 = 0
            for l in range(n):
                average0 += (self.metadata_dict[self.iter_ - l - 1]['phi'])
                average1 += (self.metadata_dict[self.iter_ - l]['phi'])
            mean0 = average0 / n
            mean1 = average1 / n
            percent_change = (mean1 - mean0) / mean0 * 100
            log.info('ALGORITHM: Log KL %i iteration rolling average percent change = %f' % (n, percent_change))
            if percent_change > 0:
                self.isConvergedCounter += 1
                if self.isConvergedCounter <= max_halfing:
                    log.info('ALGORITHM: KL rolling average increased. Dividing max stepsize by 2')
                    self.step_hyperparameter = self.step_hyperparameter / 2
                    return False
                else:
                    return True
            elif np.abs(percent_change) < tol:
                # if percent_change > 0:
                log.info('ALGORITHM: Tolerence = %f for rolling average reached' % tol)
                return True
            else:
                return False


                def getBandwidth(self, deltasMetricDeltas):
        median = np.median(np.trim_zeros(np.sqrt(deltasMetricDeltas).flatten()))
        return median ** 2 / np.log(self.nParticles + 1) # TODO REMOVE THIS


    def getDeltas(self, ensemble1, ensemble2):
        """
        Gets separation between ensembles {x}, {y}.
        Args:
            ensemble1: Ensemble {x} n x d array of particle positions
            ensemble2: Ensemble {y} n x d array of particle positions

        Returns: n1 x n2 x d array of separations.

        """
        return (ensemble1[:, np.newaxis] - ensemble2).reshape(-1, ensemble1.shape[1]).reshape((ensemble1.shape[0], ensemble2.shape[0], self.DoF))

    def getMetricDeltas(self, metric, deltas):
        """
        Contracts metric (d x d) with deltas (n1, n2, d)
        Requires n x d arrays for particles.
        Args:
            metric:
            deltas:

        Returns:

        """
        if self.iter_ == 0:
            self.contract_md = oe.contract_expression('db, mnb -> mnd', metric.shape, deltas.shape)
        return self.contract_md(metric, deltas)
        # return np.einsum('db, mnb -> mnd', metric, deltas)



    def getSteinDiscrepancy(self, kx, step):
        if step.shape[0] != self.nParticles:
            step = step.reshape(self.nParticles, self.DoF)
        if self.iter_ == 0:
            self.steinContraction = oe.contract_expression('md, mn, nd -> ', step.shape, kx.shape, step.shape)
        return self.steinContraction(step, kx, step)




    def getHessianKernel(self, h, kx, gkx, metric, metricDeltas):
        """
        Calculates the hessian of the kernel for svgd linesearch
        Args:
            h:
            kx:
            gkx:
            metric:
            metricDeltas:

        Returns: n x n x d x d array

        """
        if self.iter_ == 0:
            self.contract_gkx_md = oe.contract_expression('nmd, nmb -> nmdb', gkx.shape, metricDeltas.shape)
            self.multiply_kx_metric = oe.contract_expression('nd, ef -> ndef', kx.shape, metric.shape)
        return (-2 / h) * (self.contract_gkx_md(gkx, metricDeltas) + self.multiply_kx_metric(kx, metric))

def getDeltasMetricDeltas(self, deltas, metricDeltas):
    """
    Contracts deltas with metricDeltas. Produces metricImage.
    Requires n x d arrays for particles.
    Args:
        deltas:
        metricDeltas:

    Returns:

    """
    # TODO saves optimal contraction path. Comment this back in later. Did this to test MMD.
    if self.iter_ == 0:
        self.contract_d_md = oe.contract_expression('mnd, mnd -> mn', deltas.shape, metricDeltas.shape)
    return self.contract_d_md(deltas, metricDeltas)
    # return np.einsum('mnd, mnd -> mn', deltas, metricDeltas)

    #TODO come up with a better solution than this!
def getDeltasMetricDeltas_XY(self, deltas, metricDeltas):
    """
    FOR MMD
    Requires n x d arrays for particles.
    Args:
        deltas:
        metricDeltas:

    Returns:

    """
    # TODO saves optimal contraction path. Comment this back in later. Did this to test MMD.
    if self.iter_ == 0:
        self.contract_d_md_MMD_XY = oe.contract_expression('mnd, mnd -> mn', deltas.shape, metricDeltas.shape)
    return self.contract_d_md_MMD_XY(deltas, metricDeltas)

def getDeltasMetricDeltas_YY(self, deltas, metricDeltas):
    """
    FOR MMD
    Requires n x d arrays for particles.
    Args:
        deltas:
        metricDeltas:

    Returns:

    """
    # TODO saves optimal contraction path. Comment this back in later. Did this to test MMD.
    if self.iter_ == 0:
        self.contract_d_md_MMD_YY = oe.contract_expression('mnd, mnd -> mn', deltas.shape, metricDeltas.shape)
    return self.contract_d_md_MMD_YY(deltas, metricDeltas)
    #TODO come up with a better solution than this!

def getKernelPiecewise(self, bandwidth, deltasMetricDeltas):
    # Code copied from getDeltasMetricDeltas
    # return np.exp(-(np.einsum('mnd, mnd -> mn', deltas, metricDeltas))/bandwidth)
    return np.exp(-1 / bandwidth * (deltasMetricDeltas))

def getGradKernelPiecewise(self, bandwidth, kernel, metricDeltas):
    # Gradient with respect to first argument of kernel
    return -2 / bandwidth * kernel[..., np.newaxis] * metricDeltas

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)


        # dim = self.DoF * self.nParticles
        # Terms in H_bar
        # if self.iter_ == 0:
        #     self.contract_H_bar_1 = oe.contract_expression("xy, xy, xbd -> ybd", kernel.shape, kernel.shape, GN_Hmlpt.shape)
        #     self.contract_H_bar_2 = oe.contract_expression("xyb, xzd -> yzbd", gradKernel.shape, gradKernel.shape) # OLD HESSIAN
        # self.contract_H_bar_2 = oe.contract_expression("xyb, xyd -> ybd", gradKernel.shape, gradKernel.shape) # NEW HESSIAN
        # return ((self.contract_H_bar_1(kernel, kernel, GN_Hmlpt) + self.contract_H_bar_2(gradKernel, gradKernel)) / self.nParticles)



    def phi_new(self, T, wT, gradwT, eps):
        """
        i) Ensures that pushforward KL is smaller than original KL
        ii) Ensures T(x) = I(x) + eps * Q(x) is invertible. If the jacobian of this map is nonzero, then by inverse
         function theorem T(x) is invertible.
        Args:
            T: Particles
            wT: step
            gradwT: gradient of step
            eps: stepsize

        Returns: KL between pushforward and target.

        """
        dets = np.linalg.det(np.eye(self.DoF)[np.newaxis] + eps * gradwT)
        if np.any(np.sign(dets) != np.sign(dets[0])):
            log.info('PHI: Map T non-invertible with step %f. Returning infty' % eps)
            # pass
            return np.inf
        a = np.mean(self.getMinusLogPosterior_ensemble_new(T + eps * wT), axis=0)
        b = np.mean(np.log(np.abs(dets)), axis=0)
        log.info('PHI: Posterior average = %f, LogDet average = %f' % (a, b))
        phi = a - b
        # phi = np.mean(self.getMinusLogPosterior_ensemble_new(T + eps * wT) - np.log(np.abs(dets)), axis=0)
        if phi < 0:
            log.info('PHI: KL of pushforward is negative. Returning infty')
            return np.inf
        else:
            return phi


        # def linesearch_testingmath(self, gv, gmlpt=None):
    #     delta_phi =
    #     gv = self.getJacobianMapSVGD()
    #
    #     eps = 1
    #     eps_max = np.infty
    #     eps_min = 0
    #     def wolfe1(eps):
    #
    #
    #     wolfe2 =
    #     cond1=False
    #     cond2=False
    #     while cond1 is False or cond2 is False:
    #         if wolfe1(eps)










    def linesearch_new(self, T, wT, gradwT, step):
        """

        Args:
            T:
            wT:
            gradwT:
            step:

        Returns:

        """
        if self.iter_ == 0:
            phi_percent_change = np.inf
        maxiter = 5
        counter = 0
        phi0 = self.phi_new(T, wT, gradwT, 0)
        self.phi0 = copy.deepcopy(phi0) # Made global to be used in convergence criteria
        while True:
            phi_step = self.phi_new(T, wT, gradwT, step)
            if phi_step > phi0:
                counter += 1
                if counter != maxiter:
                    phi_percent_change = 100 * (-phi_step + phi0) / phi0
                    if phi_step != np.infty:
                        log.info('LINESEARCH: Backtracking (%i) phi_step %f > phi0 %f' % (counter, phi_step, phi0))
                    step = step / 2
                else:
                    return step / 2
            else:
                phi_percent_change = 100 * (-phi_step + phi0) / phi0
                log.info('LINESEARCH: Good step found. %f' % step)
                log.info('LINESEARCH: phi_step < phi0, %f < %f' % (phi_step, phi0))
                log.info('LINESEARCH: KL percent change between pushforward and target %f' % -phi_percent_change)
                return step


    # def mgJ_new(self, kernel, gradKernel, gmlpt):
    #     # As needed for SVGD
    #     if self.iter_ == 0:
    #         self.contract_term_mgJ = oe.contract_expression('mn, mo -> no', (kernel.shape), (gmlpt.shape))
    #     # mgJ = -1 * np.einsum('mn, mo -> no', kernel, gmlpt) / self.nParticles + np.mean(gradKernel, axis=0)
    #     mgJ = -1 * self.contract_term_mgJ(kernel, gmlpt) / self.nParticles + np.mean(gradKernel, axis=0)
    #     if self.optimizeMethod == 'SVN':
    #         mgJ = mgJ.flatten()
    #     return mgJ



def HE_laplacianKDE_term(self, Hesskx):
    if self.iter_ == 0:
        self.contract_laplacianHesskx = oe.contract_expression('mndd -> m', Hesskx.shape)
    lapKDE = self.contract_laplacianHesskx(Hesskx)
    return lapKDE

def HE_gradKx_Xi_term(self, gkx, xi):
    if self.iter_ == 0:
        # self.contract_gkx_xi = oe.contract_expression('mnd, nd -> m', gkx.shape, xi.shape)
        self.contract_gkx_xi = oe.contract_expression('mnd, nd -> m', gkx.shape, xi.shape, optimize='auto-hq')
    gkx_xi = self.contract_gkx_xi(gkx, xi)
    return gkx_xi

def HE_cost(self, hi, dict):
    h = 1 / copy.deepcopy(hi)
    M = copy.deepcopy(dict['metric'])
    mD = copy.deepcopy(dict['metricDeltas'])
    dMd = copy.deepcopy(dict['deltaMetricDeltas'])
    xi_dict = self.getXi(h, mD, dMd)
    xi = copy.deepcopy(xi_dict['xi'])
    kx = copy.deepcopy(xi_dict['kx'])
    gkx = copy.deepcopy(xi_dict['gkx'])
    hesskx = self.getHessianKernel(h, kx, gkx, M, mD)
    a = self.HE_laplacianKDE_term(hesskx)
    # b = self.HE_gradKx_Xi_term(gkx, xi) # This version makes the code not deterministic (it has a bug)
    b = np.einsum('mnd, nd -> m', gkx, xi)
    c = (a - b) ** 2
    cost = np.sum(c) * h ** 2 / (self.nParticles ** 2) # THIS ONE WORKS!!!
    #############################################
    # For debug purposes
    #############################################
    # import deepdish as dd
    # save_dict_DEBUG = {}
    # save_dict_DEBUG['dict'] = dict
    # save_dict_DEBUG['xi'] = xi
    # save_dict_DEBUG['kx'] = kx
    # save_dict_DEBUG['gkx'] = gkx
    # save_dict_DEBUG['hesskx'] = hesskx
    # save_dict_DEBUG['a'] = a
    # save_dict_DEBUG['b'] = b
    # save_dict_DEBUG['c'] = c
    # save_dict_DEBUG['cost'] = cost
    # dd.io.save(os.path.join(self.OUTPUT_DIR, 'DEBUG_ACCURACY.h5'), save_dict_DEBUG)

    return {'cost': cost, 'kx': kx, 'gkx': gkx, 'hesskx': hesskx}

def getXi(self, h, metricDeltas, deltaMetricDeltas):
    """

    Args:
        metricDeltas:
        deltaMetricDeltas:

    Returns:

    """
    kx = copy.deepcopy(self.getKernelPiecewise(h, deltaMetricDeltas))
    gkx = copy.deepcopy(self.getGradKernelPiecewise(h, kx, metricDeltas))
    # Note that summing over axis=1 is summing over second variable in k(x, y)
    xi = np.sum(gkx, 1) / np.sum(kx, 1).reshape(self.nParticles, 1)
    # Dict takes this structure to match BM_cost and HE_cost
    return {'xi': xi, 'kx': kx, 'gkx': gkx, 'hesskx': None}

    # if hi2 > 0:
    #     if cost1 < cost0:
    #         if cost2 < cost1:
    #             ret_dict = cost_dict2
    #         else:
    #             ret_dict = cost_dict1
    #     else:
    #         if cost2 < cost0:
    #             ret_dict = cost_dict2
    #         else:
    #             ret_dict = cost_dict0
    # else:
    #     if cost1 < cost0:
    #         ret_dict = cost_dict1
    #     else:
    #         ret_dict = cost_dict0
    # # assert(hi > 0)
    # return ret_dict

    def BM_cost(self, hi, dict):
        h = 1 / hi
        metricDeltas = dict['metricDeltas']
        deltaMetricDeltas = dict['deltaMetricDeltas']
        tau = 0.01
        X = dict['X']
        # np.random.seed(int(time()))
        X_BM = X + np.sqrt(2 * tau) * np.random.normal(0, 1, (self.nParticles, self.DoF))
        try:
            # cost_dict will be modified to return a dict conforming to style set in HE_cost
            cost_dict = self.getXi(h, metricDeltas, deltaMetricDeltas)
            X_KDE = X - tau * cost_dict['xi'] # wangs paper
            cost = self.getMMD(X_KDE, X_BM, mode='BM')
            cost_dict.pop('xi')
            cost_dict['cost'] = cost
            return cost_dict
        except:
            return {'cost': np.infty, 'kx': None, 'gkx': None, 'hesskx': None}

    def three_point_quadratric_optimize_NEW(self, h0, cost, dict):
        # adapted from https://github.com/YiifeiWang/Accelerated-Information-Gradient-flow/blob/master/utils/BM_bandwidth.m
        # np.random.seed(1)
        hi0 = 1 / h0
        explore_ratio = 1.1
        cost_dict0 = cost(hi0, dict)
        cost0 = cost_dict0['cost']
        cost_dict0.pop('cost')
        cost_dict0['h'] = h0
        eps = 1e-6
        # cost0eps = cost(hi0 + eps, dict)['cost'] # NOTE: this is the term that made the code non-derterministic before.
        gCost0 = (cost(hi0 + eps, dict)['cost'] - cost0) / eps
        if gCost0 < 0:
            hi1 = hi0 * explore_ratio
        else:
            hi1 = hi0 / explore_ratio
        if hi1 < 0 or hi1 == None or hi1 == 0 or np.isinf(hi1) or np.isnan(hi1):
            return cost_dict0
            # return h0
        cost_dict1 = cost(hi1, dict)
        cost1 = cost_dict1['cost']
        cost_dict1.pop('cost')
        cost_dict1['h'] = 1 / hi1
        s = (cost1 - cost0) / (hi1 - hi0)
        hi2 = (hi0 * s - 0.5 * gCost0 * (hi1 + hi0)) / (s - gCost0)
        if hi2 < 0 or hi2 == None or hi2 == 0 or np.isinf(hi2) or np.isnan(hi2):
            if cost1 < cost0:
                return cost_dict1
            else:
                return cost_dict0
        cost_dict2 = cost(hi2, dict)
        cost2 = cost_dict2['cost']
        cost_dict2.pop('cost')
        cost_dict2['h'] = 1 / hi2
        log.info('BANDWIDTH: h0 = %f, h1 = %f, h2 = %f' % (h0, 1/hi1, 1/hi2))
        log.info('BANDWIDTH: cost0 = %f, cost1 = %f, cost2 = %f' % (cost0, cost1, cost2))
        # if np.isnan(hi2) == False and hi2 > 0:
        ################################################
        # For debugging purposes
        ################################################
        # self.DEBUG_tpqo[self.iter_] = self.debug_tpqo_dict_init()
        # self.DEBUG_tpqo[self.iter_]['cost_dict0'] = cost_dict0
        # self.DEBUG_tpqo[self.iter_]['cost_dict1'] = cost_dict1
        # self.DEBUG_tpqo[self.iter_]['cost_dict2'] = cost_dict2
        # self.DEBUG_tpqo[self.iter_]['gCost0'] = gCost0
        # self.DEBUG_tpqo[self.iter_]['cost0eps'] = cost0eps
        # self.DEBUG_tpqo[self.iter_]['s'] = s
        if hi2 > 0:
            if cost1 < cost0:
                if cost2 < cost1:
                    ret_dict = cost_dict2
                else:
                    ret_dict = cost_dict1
            else:
                if cost2 < cost0:
                    ret_dict = cost_dict2
                else:
                    ret_dict = cost_dict0
        else:
            if cost1 < cost0:
                ret_dict = cost_dict1
            else:
                ret_dict = cost_dict0
        # assert(hi > 0)
        return ret_dict


def getMMD(self, X, Y, mode, deltaIDeltaX=None):
    """
    Calculates MMD. If particle pairs already calculated, use that.
    Args:
        X: ensemble 1
        Y: ensemble 2
        deltaIDeltaX: optional array if already precomputed
        mode: 'GT' ground truth, 'BW' bandwidth via brownian motion

    Returns: scalar discrepancy measure

    """
    n = X.shape[0]
    m = Y.shape[0]

    dxy = self.getDeltas(X, Y)
    dyy = self.getDeltas(Y, Y)
    bw = 1 # bandwidth for MMD kernels
    # Use MMD to calculate accuracy of samples compated to ground truth.
    if mode == 'GT':
        if type(deltaIDeltaX) is np.ndarray:
            dxx = deltaIDeltaX
        else:
            dxx = self.getDeltas(X, X)
        a = 1 / (n ** 2) * np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas(dxx, dxx))) \
            - 2 / (m * n) * np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_XY(dxy, dxy))) \
            + 1 / (m ** 2) * np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_YY(dyy, dyy)))
        return a * (2 * np.pi) ** (-1 * self.DoF / 2)
    # Use MMD in calculation of Brownian motion bandwidth selection.
    elif mode == 'BM':
        if type(deltaIDeltaX) is np.ndarray:
            dxx = deltaIDeltaX
        else:
            dxx = self.getDeltas(X, X)
        a = 1 / (n ** 2) * np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas(dxx, dxx))) \
            - 2 / (m * n) * np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas(dxy, dxy))) \
            + 1 / (m ** 2) * np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas(dyy, dyy)))
        return a * (2 * np.pi) ** (-1 * self.DoF / 2)

def check_if_same(self, A, B, tol=None):
    if tol==None:
        # tol = 1e-16
        tol = 1e-25
    return np.allclose(A, B, rtol=tol, atol=tol)




    # def getSVGD_noise_correction_IN_NOISE(self, kx, eps):
    #     dim = self.DoF * self.nParticles
    #     DK_nndd = self.getSVGD_Diffusion(kx)
    #     DK = DK_nndd.swapaxes(1, 2).reshape(dim, dim)
    #     np.random.seed(int(time()))
    #     noise_correction = np.random.normal(0, 2 * eps * DK, dim)
    #     return_dict = {'noise_correction': noise_correction.reshape(self.nParticles, self.DoF), 'diffusion': DK_nndd}
    #     return return_dict


    ##########################################################
    # Stochastic SVN: Defunct methods to remove later!!!
    ##########################################################
    # This works separately (for the whole hessian)
    # def gradH_action_mnbd(self, gradH, Bnndd):
    #     return np.einsum('mnabc, nmbc -> ma', gradH, Bnndd)
    # def gradH_action_BD(self, gradH, Bndd):
    #     if self.iter_ == 0:
    #         self.gradH_action_BD_contract = oe.contract_expression('mabc, mbc -> ma', gradH.shape, Bndd.shape)
    #     return self.gradH_action_BD_contract(gradH, Bndd)
    def gradK_action_BD(self, gkx, Bndd):
        vec_smart_c = np.einsum('nbd, nmd -> mb', Bndd, gkx) / self.nParticles
        return vec_smart_c
    #******************************************************
    #******************************************************
    #******************************************************










    # def gradH_BD(self, Hmlpt, kx, gkx, hesskx):
    #     if self.iter_ == 0:
    #         self.gradH_bar_a = oe.contract_expression('xab, xy, xyj -> yabj', Hmlpt.shape, kx.shape, gkx.shape)
    #         self.gradH_bar_b = oe.contract_expression('xyaj, xyb -> yabj', hesskx.shape, gkx.shape)
    #     # For gradient on second variable
    #     return (-1 * self.gradH_bar_a(Hmlpt, kx, gkx) - self.gradH_bar_b(hesskx, gkx)) / self.nParticles


    #######################################################
    # Stochastic SVN : Reduced memory methods
    #######################################################



    # def getK_action_mat(self, kx, A_nndd):
    #     if self.iter_ == 0:
    #         self.contract_K_mat_action = oe.contract_expression('mn, nodb -> modb', (kx.shape), (A_nndd.shape))
    #     return self.contract_K_mat_action(kx, A_nndd) / self.nParticles






    # def K_action_BD(self, kx, HBD_inv):
    #     if self.iter_ == 0:
    #         self.K_HBD_inv_contract = oe.contract_expression('nn, nbd, -> nbd', kx.shape, HBD_inv.shape)
    #     return self.K_HBD_inv_contract(kx, HBD_inv)



    # def gradK_action_BD(self, gkx, Bnndd):
    #     """
    #     Gets the action of gradient of the augmented K on an nndd matrix
    #     Args:
    #         gkx: Gradient of the kernel. n x n x d shaped
    #         A: n x n x d x d shaped matrix
    #
    #     Returns: Action array with shape n x d
    #
    #     """
    #     vec_smart_c = np.einsum('nnbd, nmd -> mb', Bnndd, gkx) / self.nParticles
    #     return vec_smart_c
    #######################################################
    # Stochastic SVN : Preliminary calculations
    #######################################################
    # def gradH(self, Hmlpt, kx, gkx, hesskx):
    #     if self.iter_ == 0:
    #         self.gradH_bar_a = oe.contract_expression('xab, xy, xzj -> yzabj', Hmlpt.shape, kx.shape, gkx.shape)
    #         self.gradH_bar_b = oe.contract_expression('xzaj, xyb -> yzabj', hesskx.shape, gkx.shape)
    #     # For gradient on second variable
    #     return (-1 * self.gradH_bar_a(Hmlpt, kx, gkx) - self.gradH_bar_b(hesskx, gkx)) / self.nParticles


    #######################################################
    # Stochastic SVN : Main calculations
    #######################################################
    # def getSVN_Diffusion(self, kx, K_H_inv):
    #     H_inv_K = np.einsum('nmdb -> mnbd', K_H_inv)
    #     return self.getK_action_mat(kx, H_inv_K)
    # def getSVN_Diffusion_BD(self, kx, K_H_inv):
    #     H_inv_K = np.einsum('nmdb -> mnbd', K_H_inv)
    #     return self.getK_action_mat(kx, H_inv_K)


    # def getDivergenceDiffusion(self, gkx, K_H_inv, gradH):
    #     H_inv_K = np.einsum('nmdb -> mnbd', K_H_inv)
    #     divK = np.mean(gkx, axis=0)
    #     tmp_vec = self.gradH_action(gradH, H_inv_K) # output is an n x d array
    #     a = self.gradK_action(gkx, H_inv_K)
    #     b = np.einsum('mndb, nb -> md', K_H_inv, tmp_vec)
    #     c = np.einsum('nmdb, mb -> nd', K_H_inv, divK)
    #     return a - b + c

    # def getNoiseSVN(self, kx, H_inv_sqrt):
    #     B = np.random.normal(0, 1, self.dim)
    #     tmp = (np.sqrt(2) * np.einsum('ij, j -> i', H_inv_sqrt, B)).reshape(self.nParticles, self.DoF)
    #     return self.getK_action_vec(kx, tmp) # requires a  n x d vector

    ################################################################################################################


def findPosdefEigendecompisition(self, Hndnd):
    tol = 1e-8
    eigvals, eigvecs = np.linalg.eigh(Hndnd)
    jitter = np.abs(np.min(eigvals)) + tol
    while np.any(eigvals < 0):
        log.info('EIGENDECOMPOSE: Hessian not posdef. Adding jitter = %f.' % jitter)
        eigvals, eigvecs = np.linalg.eigh(Hndnd + jitter * np.eye(self.dim, self.dim))
        if np.all(eigvals > 0):
            return eigvals,eigvecs
        # jitter = jitter * 10
        jitter *= 1.1
    # raise Exception('EIGENDECOMPOSE: Could not make posdef through perturbation.')
    else:
        return eigvals,eigvecs

def compute_sqrt_if_possible(self, x):
    size = x.shape[0]
    sqrt = scipy.linalg.sqrtm(x)
    jitter = 1e-9
    if sqrt.dtype == 'complex128':
        while jitter < 1.0:
            log.info('SQRT: Matrix complex! Adding Jitter')
            sqrt = scipy.linalg.sqrtm(x + jitter * np.eye(size))
            if sqrt.dtype != 'complex128':
                return sqrt
            jitter = jitter * 10
        raise Exception('Cholesky factorization failed.')
    else:
        return sqrt

def nearestPD(self, A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if self.isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not self.isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3
    def sparse_cholesky(self, A): # The input matrix A must be a sparse symmetric positive-definite.

        n = A.shape[0]
        LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition

        if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
            return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )
        else:
            sys.exit('The matrix is not positive definite')

    # def compute_cholesky_if_possible(self, x, jitter):
    #     size = x.shape[0]
    #     return_dict = {'sqrt': None, 'alpha': None}
    #     try:
    #         cholesky = np.linalg.cholesky(x)
    #         return_dict['sqrt'] = cholesky
    #         return_dict['alpha'] = 0
    #         return return_dict
    #     except Exception:
    #         # jitter = 1e-9
    #         while jitter < 1.0:
    #         # while jitter < 1000.0:
    #             try:
    #                 cholesky = np.linalg.cholesky(x + jitter * np.eye(size))
    #                 return_dict['sqrt'] = cholesky
    #                 return_dict['alpha'] = jitter
    #                 return return_dict
    #             except Exception:
    #                 log.info('CHOLESKY: Matrix not positive-definite! Adding alpha = %f' % jitter)
    #                 jitter = jitter * 10
    #         raise Exception('Cholesky factorization failed.')
    # def getSVGD_noise_correction(self, kx, eps):
    #     """
    #     Calculates the noise correction for SVGD. Specifically, the last term in [Eq 6, Nusken 2021]
    #     Args:
    #         kx:
    #
    #     Returns:
    #
    #     """
    #     DK = self.getSVGD_Diffusion(kx)
    #     sqrt_two_DK = np.sqrt(2 * eps * DK)
    #     np.random.seed(int(time()))
    #     brownian_noise = np.random.normal(0, 1, (self.nParticles, self.DoF))
    #     noise_correction = np.einsum('nmdb, mb -> nd', sqrt_two_DK, brownian_noise)
    #     results = {'noise_correction': noise_correction, 'diffusion': DK}
    #     return results

    # def mgJ_NOISE_CORRECTED(self, kernel, gradKernel, gmlpt, eps):
    #     if self.iter_ == 0:
    #         self.contract_term_mgJ = oe.contract_expression('mn, mo -> no', (kernel.shape), (gmlpt.shape))
    #     noise_correction = self.getSVGD_noise_correction(kernel)['noise_correction']
    #     original_direction = self.mgJ_new(kernel, gradKernel, gmlpt)
    #     mgJ_noise_corrected = eps * original_direction + np.sqrt(eps) * noise_correction
    #     return mgJ_noise_corrected

def BM_cost_NOISY_SVGD(self, hi, dict):
    """
    Proof of concept. Noisy SVGD bandwidth selection.
    Args:
        hi:
        dict:

    Returns:

    """
    ########################################################
    # Definitions
    ########################################################
    h = 1 / hi
    metricDeltas = dict['metricDeltas']
    deltaMetricDeltas = dict['deltaMetricDeltas']
    tau = 0.01
    X = dict['X']
    h_prev = dict['h_prev']
    # h_prev = h
    kx = self.getKernelPiecewise(h_prev, deltaMetricDeltas)
    gkx = self.getGradKernelPiecewise(h_prev, kx, metricDeltas)
    ########################################################
    # Get noisy step and Brownian motion update
    ########################################################
    noise_correction = self.getSVGD_noise_correction_MANUAL(kx, tau)['noise_correction']
    gamma = np.mean(gkx, axis=0)
    X_BM = X + tau * gamma + noise_correction
    ################################################################
    # Deterministic update
    ################################################################
    # SVGD_Diffusion_K = self.getSVGD_Diffusion(kx)
    cost_dict = self.getXi(h, metricDeltas, deltaMetricDeltas)
    xi = cost_dict['xi']
    if self.iter_ == 0:
        self.contract_term_mgJ = oe.contract_expression('mn, mo -> no', (kx.shape), (xi.shape))
    diffusionGradLogRho = self.contract_term_mgJ(kx, xi) / self.nParticles
    # diffusionGradLogRho = np.einsum('nmdb, mb -> nd', SVGD_Diffusion_K, xi)
    X_det = X - tau * diffusionGradLogRho
    #############################################
    # Evaluating cost function
    #############################################
    try:
        cost = self.getMMD(X_det, X_BM, mode='BM')
        cost_dict.pop('xi')
        cost_dict['cost'] = cost
        cost_dict['noise_correction'] = noise_correction
        return cost_dict
    except:
        log.info('BANDWIDTH: Exception thrown while calculating cost.')
        return {'cost': np.infty, 'kx': None, 'gkx': None, 'hesskx': None}

def getPosTrimEigendecomp(self, H_bar):
    eigvals, eigvecs = scipy.linalg.eigh(H_bar)
    eigvals = np.abs(eigvals)
    return eigvals, eigvecs


# action ACTION Action methods
####################################################
# Stochastic SVN Block Diagonal: Action methods (5)
####################################################

# # (1) //////////////////////////
# def K_action_mb(self, kx, v_nd):
#     if self.iter_ == 0:
#         self.K_action_vec_contract = oe.contract_expression('xy, yd -> xd', kx.shape, v_nd.shape)
#     return self.K_action_vec_contract(kx, v_nd) / self.nParticles
#
# # (2) ///////////////////////////
# def K_action_mbd(self, kx, Bndd):
#     return np.einsum('nm, nbd -> mnbd', kx, Bndd) / self.nParticles
#
# # (3) /////////////////////////////
# def K_action_mnbd(self, kx, Bnndd):
#     if self.iter_ == 0:
#         self.K_action_mnbd_mat_contraction = oe.contract_expression('mo, onbd -> mnbd', kx.shape, Bnndd.shape)
#     return self.K_action_mnbd_mat_contraction(kx, Bnndd) / self.nParticles
#
# # (4) //////////////////////////////
# def gradK_action_mnbd(self, gkx, mnbd):
#     if self.iter_ == 0:
#         self.gradK_mnbd_new_contraction_a = oe.contract_expression('nnbd, nmd -> mb', mnbd.shape, gkx.shape)
#         self.gradK_mnbd_new_contraction_b = oe.contract_expression('mnb, nmdb -> md', gkx.shape, mnbd.shape)
#     return (self.gradK_mnbd_new_contraction_a(mnbd, gkx) + self.gradK_mnbd_new_contraction_b(gkx, mnbd)) / self.nParticles
#
# # (5) ////////////////////////////////////////////
# def grad_hij_BD_action_mnbd(self, grad_hij_BD, Bnndd):
#     if self.iter_ == 0:
#         self.grad_hij_BD_action_mnbd_contraction = oe.contract_expression('mnikj, mnij -> mk', grad_hij_BD.shape, Bnndd.shape)
#     return self.grad_hij_BD_action_mnbd_contraction(grad_hij_BD, Bnndd)
#
# ##################################################################
# # Stochastic SVN Block Diagonal: Matrix-vector product methods (2)
# ##################################################################
#
# # (1) /////////////////////////////
# def mnbd_mb_matvec(self, mat, vec):
#     return np.einsum('mnbd, nd -> mb', mat, vec)
#
# # (2) ////////////////////////////
# def mbd_mb_matvec(self, mat, vec):
#     return np.einsum('ndb, nb -> nd', mat, vec)

#################################################################
# N x N block diagona methods
#################################################################
# def HBD_NxN(self, kx, gkx, Hmlpt):
#     return (np.einsum('Njj, Nm, Nn -> jmn', Hmlpt, kx, kx) + np.einsum('Nnj, Nmj -> jmn', gkx, gkx)) / self.nParticles




# def getSVN_direction(self, kx, gkx, gmlpt, A):
#     kbar = np.mean(gkx, axis=0)
#     return np.einsum('nmji, no, oj -> mi', A, kx, -1 * gmlpt) \
#              + self.nParticles * np.einsum('nmji, nj -> mi', A, kbar)

# (1) ////////////////////////////////////////////
# def grad_hij_BD_TESTING(self, kx, gkx, Hesskx, GN_Hmlpt, grad_GN_Hmlpt): # There is obviously room for improvement in this calculation. However, I want to make sure it works first.
#     # Term (i)
#     a_i = np.einsum('mn, mne, nij -> mije', 2 * kx, gkx, GN_Hmlpt) # Minus goes away because Hmlpt
#     b_i = np.einsum('mnie, mnj -> mije', Hesskx, gkx)
#     c_i = np.einsum('mni, mnje -> mije', gkx, Hesskx)
#     i = (a_i + b_i + c_i) / self.nParticles
#     # Term (ii)
#     a_ii = np.einsum('mn, mne, nij -> mnije', 2 * kx, gkx, -1 * GN_Hmlpt)
#     b_ii = np.einsum('mn, nije -> mnije', kx ** 2, grad_GN_Hmlpt)
#     c_ii = -1 * np.einsum('mnie, mnj -> mnije', Hesskx, gkx)
#     d_ii = -1 * np.einsum('mni, mnje -> mnije', gkx, Hesskx)
#     ii = (a_ii + b_ii + c_ii + d_ii) / self.nParticles
#     # Adds (i) along the diagonal of (ii)
#     ii[range(self.nParticles), range(self.nParticles)] += i
#     return ii

# (2) ///////////////////////////////////////////////////////
# def getSVN_Direction(self, kx, gkx, gmlpt_new, K_HBD_inv, HBD_inv_K):
#     D_SVN = self.nParticles * self.K_action_mnbd(kx, HBD_inv_K)
#     divK = np.mean(gkx, axis=0)
#     a = self.mnbd_mb_matvec(D_SVN, -1 * gmlpt_new)
#     b = self.mnbd_mb_matvec(K_HBD_inv, divK) * self.nParticles
#     return a + b

# (3) ///////////////////////////////////////////////////
# def getSVN_deterministic_correction(self, gkx, grad_hij_BD, HBD_inv_K, K_HBD_inv):
#     gradH_action_HBD_inv_K = self.grad_hij_BD_action_mnbd(grad_hij_BD, HBD_inv_K)
#     # gradH_action_HBD_inv_K = self.gradHBD_action_mnbd(grad_hij_BD, HBD_inv_K)
#     a = self.gradK_action_mnbd(gkx, HBD_inv_K)
#     b = -1 * self.mnbd_mb_matvec(K_HBD_inv, gradH_action_HBD_inv_K)
#     return (a + b) * self.nParticles

# (4) ////////////////////////////////////////////////////////////////////////////
# def getSVN_stochastic_correction(self, kx, HBD_inv_sqrt, B=None):
#     if B is None:
#         B = np.random.normal(0, 1, (self.nParticles, self.DoF))
#     # B = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim)).reshape(self.nParticles, self.DoF)
#     p = self.mbd_mb_matvec(HBD_inv_sqrt, B)
#     return np.sqrt(2 * self.nParticles) * self.K_action_mb(kx, p)

# def getSVGD_stochastic_correction(self, sqrt_kx, B_bm=None):
#     # np.random.seed(self.iter_)
#     if B_bm is None:
#         B_bm = np.random.normal(0, 1, (self.DoF, self.nParticles))
#         # B_bm = B.reshape(self.DoF, self.nParticles)
#     return np.einsum('mn, dn -> dm', sqrt_kx, B_bm).reshape(self.nParticles, self.DoF, order='C')
# log.info('INFO: FIRST RANDOM ENTRY %F' % B[0])
# Bnd = B.reshape(self.DoF, self.nParticles)
# return (np.einsum('db,mb -> dm', sqrt_kx, B).flatten()).reshape(self.nParticles, 2, order='F')
# return (np.einsum('mn, bn -> bm', sqrt_kx, Bnd).flatten()).reshape(self.nParticles, self.DoF)


############################################################
# NEW STOCHASTIC SVN METHODS
############################################################
# def getDeterministicCorrection_a(self, gkx, A):
#     return np.einsum('mne, nmie -> mi', gkx, A) \
#              - np.einsum('mne, nnie -> mi', gkx, A)
# def getDeterministicCorrection_b(self, kx, gkx, Hesskx, Hmlpt, gradHmlpt, A):
#     a_tmp = np.einsum('oije, om, on -> mnoije', gradHmlpt, kx, kx) \
#             + np.einsum('oij, ome, on -> mnoije', Hmlpt, gkx, kx) \
#             + np.einsum('oij, om, one -> mnoije', Hmlpt, kx, gkx) \
#             + np.einsum('onie, omj -> mnoije', Hesskx, gkx) \
#             + np.einsum('oni, omje -> mnoije', gkx, Hesskx)
#
#     a = np.einsum('mnoije, mpif, noje -> pf', a_tmp, A, A)
#
#     # This guy requires a bit of modification
#     b_tmp = np.einsum('Nij, Noe, Nn -> noije', -1 * Hmlpt, gkx, kx) \
#             - np.einsum('Nni, Noje -> noije', gkx, Hesskx)
#     b = np.einsum('noije, opif, noje -> pf', b_tmp, A, A)
#
#     c_tmp = np.einsum('Nij, Nm, Nne -> mnije', -1 * Hmlpt, kx, gkx) \
#             - np.einsum('Nnie, Nmj -> mnije', Hesskx, gkx)
#     c = np.einsum('mnije, mpif, nnje -> pf', c_tmp, A, A)
#
#     return a + b + c

# def getDeterministicCorrection_simplemath(self, kx, gkx, Hesskx, Hmlpt, gradHmlpt, A):
#     i = np.einsum('mne, nmie -> mi', gkx, A) - np.einsum('mne, nnie -> mi', gkx, A)
#
#     a_tmp = np.einsum('oije, om, on -> mnoije', gradHmlpt, kx, kx) \
#             + np.einsum('oij, ome, on -> mnoije', Hmlpt, gkx, kx) \
#             + np.einsum('oij, om, one -> mnoije', Hmlpt, kx, gkx) \
#             + np.einsum('onie, omj -> mnoije', Hesskx, gkx) \
#             + np.einsum('oni, omje -> mnoije', gkx, Hesskx)
#
#     a = np.einsum('mnoije, mpif, noje -> pf', a_tmp, A, A)
#
#     # This guy requires a bit of modification
#     b_tmp = np.einsum('Nij, Noe, Nn -> noije', -1 * Hmlpt, gkx, kx) \
#             - np.einsum('Nni, Noje -> noije', gkx, Hesskx)
#     b = np.einsum('noije, opif, noje -> pf', b_tmp, A, A)
#
#     c_tmp = np.einsum('Nij, Nm, Nne -> mnije', -1 * Hmlpt, kx, gkx) \
#             - np.einsum('Nnie, Nmj -> mnije', Hesskx, gkx)
#     c = np.einsum('mnije, mpif, nnje -> pf', c_tmp, A, A)
#
#     return a + b + c + i

# def getSVGD_Diffusion(self, kx):
#     """
#     Calculates the n x n x d x d diffusion matrix as in [Eq 6, Nusken 2021] for SVGD
#     Args:
#         kx:
#
#     Returns:
#
#     """
#     DK_nndd = np.einsum('ij, db -> ijdb', kx, np.eye(self.DoF)) / self.nParticles # agreed
#     return DK_nndd

# def pos_and_trim_mat(self, A):
#     """
#     Flip negative eigenvalues and remove all eigenvalues with small values.
#     Args:
#         A:
#
#     Returns:
#
#     """
#     tolerence = 1e-10
#     eigsystem = np.linalg.eigh(A)
#     eigvalues = eigsystem[0]
#     eigvectors = eigsystem[1].T
#     max_eigenvalue = np.max(eigvalues)
#     truncate_index = np.argmax(eigvalues > tolerence * max_eigenvalue)
#     truncated_eigenvalues = copy.deepcopy(eigvalues[truncate_index:])
#     truncated_eigenvectors = copy.deepcopy(eigvectors[truncate_index:])
#     sqrtD = np.diag(np.sqrt(truncated_eigenvalues))
#     return truncated_eigenvectors.T @ sqrtD @ truncated_eigenvectors

# def getSymmetricSqrt(self, kx):
#     """
#     Calculates the low rank version of square root.
#     Args:
#         kx:
#
#     Returns:
#
#     """
#     tolerence = 1e-10
#     eigsystem = np.linalg.eigh(kx)
#     eigvalues = eigsystem[0]
#     eigvectors = eigsystem[1].T
#     max_eigenvalue = np.max(eigvalues)
#     truncate_index = np.argmax(eigvalues > tolerence * max_eigenvalue)
#     truncated_eigenvalues = copy.deepcopy(eigvalues[truncate_index:])
#     truncated_eigenvectors = copy.deepcopy(eigvectors[truncate_index:])
#     sqrtD = np.diag(np.sqrt(truncated_eigenvalues))
#     return truncated_eigenvectors.T @ sqrtD @ truncated_eigenvectors

# def getSVGD_noise_correction_NEW(self, kx, eps):
#     """
#     Calculates noise correction to SVGD
#     Args:
#         kx:
#         eps:
#
#     Returns:
#
#     """
#     sqrt_K = self.getSymmetricSqrt(kx)
#     np.random.seed(int(time()))
#     brownian_noise = np.random.normal(0, 1, (self.nParticles, self.DoF))
#     contraction = (sqrt_K @ brownian_noise).flatten(order='F').reshape(self.nParticles, self.DoF)
#     noise_correction = np.sqrt(2 * eps / self.nParticles) * contraction
#     return_dict = {'noise_correction': noise_correction}
#     return return_dict

# def getSVGD_noise_correction_MANUAL(self, kx, eps):
#     dim = self.DoF * self.nParticles
#     DK_nndd = self.getSVGD_Diffusion(kx) # Put the 2 in here to make the matrix more numerically stable
#     DK = DK_nndd.swapaxes(1, 2).reshape(dim, dim)
#     # ldl = scipy.linalg.ldl(2 * eps * DK)
#     # sqrt_DK = ldl[0] @ np.sqrt(ldl[1])
#     sqrt_DK = self.compute_cholesky_if_possible(DK)
#     # sqrt_DK = scipy.linalg.sqrtm(self.nearestPD(2 * eps * DK))
#     # sqrt_DK = self.compute_cholesky_if_possible(2 * eps * DK)
#     np.random.seed(int(time()))
#     brownian_noise = np.random.normal(0, 1, (self.nParticles * self.DoF))
#     noise_correction = np.sqrt(2 * eps / self.nParticles) * sqrt_DK @ brownian_noise
#     return_dict = {'noise_correction': noise_correction.reshape(self.nParticles, self.DoF), 'diffusion': DK_nndd}
#     return return_dict



# Commented out to rename
# def grad_w_new(self, alphas, gradKernel):
#     # -1 here because gradient is taken w.r.t second argument for this expression!
#     if self.iter_ == 0:
#         self.contract_terms_gradw = oe.contract_expression('nd, nyb -> ydb', (alphas.shape), (gradKernel.shape))
#     # return -1 * np.einsum('nd, nyb -> ydb', alphas, gradKernel)
#     return -1 * self.contract_terms_gradw(alphas, gradKernel)

# def get_v_SVN(self, alphas, kernel):
#     if self.iter_ == 0:
#         self.contract_term_w = oe.contract_expression('xd, xn -> nd', (alphas.shape), (kernel.shape))
#     return self.contract_term_w(alphas, kernel)
#     # return np.einsum('xd, xn -> nd', alphas, kernel)

# test suite TEST Test

# *** Full calculations ***
# cholesky_dict = self.stein.compute_cholesky_if_possible(self.H_bar, 1e-9)
# self.alpha = cholesky_dict['alpha']
# self.H_PD_ndnd = self.H_bar + self.alpha * np.eye(self.dim)
# self.H_inv_ndnd = np.linalg.inv(self.H_PD_ndnd)
# self.H_inv_nndd = self.stein.reshapeNDNDtoNNDD(self.H_inv_ndnd)
# self.A_ndnd = self.H_inv_ndnd @ self.K
# self.A_nndd = self.stein.reshapeNDNDtoNNDD(self.A_ndnd)
# self.D_SVN = self.nParticles * self.K @ self.H_inv_ndnd @ self.K
# self.phi = lambda m, d: m * self.DoF + d
# self.gradH = np.einsum('ijk -> kij', nd.Gradient(self.form_H_for_numdiff)(self.X.flatten()))

# ######################################
# # Testing BD SVN-Hessian
# ######################################
# def noise_sampling_plots(self):
#     nsamples = 1000
#     # B = np.random.multivariate_normal(np.zeros(self.dim), self.HBD_inv_ndnd).reshape(self.nParticles, self.DoF)
#     samples = np.random.multivariate_normal(np.zeros(self.dim), self.HBD_inv_ndnd, nsamples)
#     test = samples.reshape(nsamples * self.nParticles, self.DoF)
#     x = test[:, 0]
#     y = test[:, 1]
#     for m in range(nsamples):
#         plt.scatter(x[m * self.nParticles : self.nParticles * (m + 1)], y[m * self.nParticles : self.nParticles * (m + 1)])
#     plt.show()
#     pass

# (2a)
# def test_getSVN_Deterministic_correction(self): # Algorithm for full SVN-Hessian
#     # Check if implementation matches math in eq (45)
#     test_a = self.stein.getSVN_Deterministic_correction(self.kx, self.gkx, self.Hesskx, self.GN_Hmlpt, self.grad_GN_Hmlpt, self.A_nndd)
#     test_b = self.nParticles * (np.einsum('abc, bc -> a', self.gradK, self.A_ndnd)
#                                 - np.einsum('ab, bce, ce -> a', self.A_ndnd.T, self.gradH, self.A_ndnd)).reshape(self.nParticles, self.DoF)
#     assert np.allclose(test_a, test_b, rtol=1e-6, atol=1e-6)


    # # Fixing low rank of SVN-Hessian
    # scale = 0.5
    # bad_particles = np.argwhere(tf.linalg.matrix_rank(HBD) < self.DoF).squeeze()
    # while bad_particles.size > 0:
    #     try:
    #         HBD[bad_particles] += np.eye(self.DoF)[np.newaxis, ...] * scale
    #     except:
    #         HBD[bad_particles] += np.eye(self.DoF) * scale
    #     good_particles_i = np.argwhere(tf.linalg.matrix_rank(HBD[bad_particles]) == self.DoF).squeeze()
    #     bad_particles = np.delete(bad_particles, good_particles_i)
    #
    # # Test kernel rank
    # if tf.linalg.matrix_rank(kx) < self.nParticles:
    #     raise Exception('Rank deficient kernel!')
    # #
    bad_particles = np.argwhere(tf.linalg.matrix_rank(HBD) < self.DoF)
    if bad_particles.size > 0:
        log.info('WARNING: SVN-Hessian rank deficient')
        print(bad_particles.T)
         raise Exception('Rank deficient SVN-Hessian!'

         # print('stochastic norm', np.linalg.norm(v_stc.flatten()), 'det norm', np.linalg.norm(v_svn.flatten()))
         # LHBDop = tf.linalg.LinearOperatorFullMatrix(LHBD, is_self_adjoint=True, is_positive_definite=True)
         # v_stc = tf.linalg.experimental.conjugate_gradient(LHBDop, tf.constant(v_svgd_stc), max_iter=10).x.numpy()
         # print('Thermal noise norm:', np.linalg.norm(v_stc.flatten()))
         # print('Force norm:', np.linalg.norm(v_svn.flatten()))


        elif method == 'sSVN_lin':
        Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
        kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0))
        kx_sqrt = tf.linalg.sqrtm(kx).numpy()
        gkx_test = contract('mni -> min', gkx)
        start1 = time()
        zero = tf.linalg.LinearOperatorZeros(self.DoF, self.DoF, [self.nParticles], dtype='float64')
        tmp1 = tf.linalg.LinearOperatorLowRankUpdate(zero, gkx_test)
        tmp1.to_dense()
        end1 = time()
        start2 = time()
        contract('mni, mnj -> mij', gkx, gkx)
        end2 = time()
        print(end1 - start1)
        print(end2 - start2)
        HBD = self.h_ij_BD(Hmlpt_new, kx, gkx)
        LHBD = tf.linalg.cholesky(HBD).numpy()
        UHBD = contract('mij -> mji', LHBD)
        v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
        if iter_ < 5:
            HBDop = tf.linalg.LinearOperatorFullMatrix(HBD, is_self_adjoint=True, is_positive_definite=True)
        v_svn = tf.linalg.experimental.conjugate_gradient(HBDop, tf.constant(v_svgd), max_iter=1).x.numpy()
        else:
        v_svn = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis])) # agreed
        B = np.random.normal(0, 1, (self.nParticles, self.DoF))
        noise = np.squeeze(tf.linalg.triangular_solve(UHBD, B[..., np.newaxis])) # agreed
        v_stc = np.sqrt(2) * contract('mn, in -> im', kx_sqrt / self.nParticles, noise.reshape(self.DoF, self.nParticles)).reshape(self.nParticles, self.DoF)
        update = v_svn * eps + v_stc * np.sqrt(eps)



        elif method == 'sSVNb':
        Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
        kx, gkx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0))
        print(np.linalg.matrix_rank(kx))
        HBD = self.h_ij_BD(Hmlpt_new, kx, gkx)
        LHBD = tf.linalg.cholesky(HBD).numpy()
        UHBD = contract('mij -> mji', LHBD)
        v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
        v_svn_a = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis])) # agreed
        v_svn = contract('mn, ni -> mi', kx, v_svn_a)
        B = np.random.normal(0, 1, (self.nParticles, self.DoF))
        noise = np.squeeze(tf.linalg.triangular_solve(UHBD, B[..., np.newaxis])) # agreed
        v_stc = np.sqrt(2 / self.nParticles) * contract('mn, ni -> mi', kx, noise)
        update = v_svn * eps + v_stc * np.sqrt(eps)


        if method == 'sSVNv0':
            Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
        kx, gkx, hesskx = getKernelWithDerivatives(X_new, h=h, get_hesskx=True)
        # kx, gkx, hesskx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0), get_hesskx=True)
        HBD = self.h_ij_BD(Hmlpt_new, kx, gkx)
        HBD_inv = tf.linalg.inv(HBD).numpy()
        HBD_inv_sqrt = tf.linalg.cholesky(HBD_inv).numpy()
        grad_GN_Hmlpt = self.getGradientGNHessianMinusLogPosterior_ensemble(X_new)
        A = self.getA_BD(HBD_inv, kx)
        v_svn = self.getSVN_direction(kx, gkx, gmlpt_new, A)
        v_det = self.getSVN_BD_Deterministic_correction(kx, gkx, hesskx, Hmlpt_new, grad_GN_Hmlpt, A)
        v_stc = self.getSVN_BD_Stochastic_correction(kx, HBD_inv_sqrt)
        update = (v_svn + v_det) * eps + v_stc * np.sqrt(eps)
        elif method == 'sSVNv1':
        Hmlpt_new = self.getGNHessianMinusLogPosterior_ensemble_new(X_new)
        # kx, gkx, hesskx = getKernelWithDerivatives(X_new, h=h, get_hesskx=True)
        kx, gkx, hesskx = getKernelWithDerivatives(X_new, h=h, M=np.mean(Hmlpt_new, axis=0), get_hesskx=True)
        HBD = self.h_ij_BD(Hmlpt_new, kx, gkx)
        ranks = np.linalg.matrix_rank(HBD)
        if np.any(ranks<self.DoF):
            raise Exception('Rank deficient svn hessian!')
        print(np.min(ranks))
        grad_GN_Hmlpt = self.getGradientGNHessianMinusLogPosterior_ensemble(X_new)
        LHBD = tf.linalg.cholesky(HBD).numpy()
        A = self.get_A_BD_stable(LHBD, kx) # Calculated without inverses
        # v_svgd = self.getSVGD_direction(kx, gkx, gmlpt_new)
        # v_svn = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis]))
        v_svn = self.getSVN_direction(kx, gkx, gmlpt_new, A)
        v_det = self.getSVN_BD_Deterministic_correction(kx, gkx, hesskx, Hmlpt_new, grad_GN_Hmlpt, A)
        v_stc = self.getSVN_BD_Stochastic_correction_stable(LHBD, kx) # Calculated without inverses
        update = (v_svn + v_det) * eps + v_stc * np.sqrt(eps)



    # DEBUG
    # v_svn_test = np.zeros((self.nParticles, self.DoF))

    # for n in range(self.nParticles):
    #     v_svn_test[n] = np.squeeze(tf.linalg.cholesky_solve(LHBD[n], v_svgd[n][..., np.newaxis]).numpy())
    # for n in range(self.nParticles):
    # v_svn_test[n] = scipy.sparse.linalg.cg(HBD[n], v_svgd[n], maxiter=1)[0]
    # self.getMinimumPerturbationCholesky(kx)

    # min_rank = np.min(tf.linalg.matrix_rank(Hmlpt_new))
    # if iter_ == 19:
    #     print('lala')
    # if min_rank == 0:
    #     print('defective hessian')
    # print('min rank of kernel repulsion', np.min(tf.linalg.matrix_rank(contract('mni, mnj -> mij', gkx, gkx))))
    # print('min rank of gn hess', np.min(tf.linalg.matrix_rank(Hmlpt_new)))
    # kx_sqrt = tf.linalg.sqrtm(kx).numpy()
    # v_svn = tf.linalg.experimental.conjugate_gradient(HBDop, tf.constant(v_svgd), max_iter=5, preconditioner=tf.linalg.LinearOperatorFullMatrix(LHBD)).x.numpy()
    # B = np.random.normal(0, 1, (self.nParticles, self.DoF))
    # noise = np.squeeze(tf.linalg.triangular_solve(UHBD, B[..., np.newaxis])) # old
    # noise = np.squeeze(tf.linalg.triangular_solve(UHBD, B[..., np.newaxis], lower='False')) # new
    # v_stc = np.sqrt(2) * contract('mn, in -> im', kx_sqrt / self.nParticles, noise.reshape(self.DoF, self.nParticles)).reshape(self.nParticles, self.DoF) # OLD
    # v_stc = np.sqrt(2 / self.nParticles) * contract('mn, in -> im', kx_sqrt, noise.flatten(order='F').reshape(self.DoF, self.nParticles)).flatten(order='F').reshape(self.nParticles, self.DoF) # NEW
    # if np.isnan(v_stc) is True:
    #     raise ValueError('noise is Nan')


