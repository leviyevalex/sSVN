from scipy import spatial
import numpy as np
from opt_einsum import contract
import timeit
import scipy

def getPairwiseDisplacement(X, Y):
    return X[:,np.newaxis,:] - Y[np.newaxis,:,:]
def getPairwiseDistance(X, Y, metric='euclidean'):
    return spatial.distance.cdist(X, Y, metric=metric)

class Gaussian_identity():
    def __init__(self, h=0.01):
        self.h = h
    #######################################################
    # Kernel functions for single observations
    #######################################################
    def k(self, x, y):
        quad_form = np.dot(x-y, x-y) / self.h
        return np.exp(-quad_form)

    def gk(self, x, y):
        return -2 / self.h * self.k(x,y) * (x-y)

    def hessk(self, x, y):
        DoF = x.size
        return -2 / self.h * (np.outer(self.gk(x,y), x-y) + self.k(x, y) * np.eye(DoF))

    #######################################################
    # Kernel functions for ensembles
    #######################################################
    def kx(self, X, Y):
        return np.exp(-getPairwiseDistance(X, Y, metric='sqeuclidean') / self.h)

    def gkx(self, X, Y):
        displacements = getPairwiseDisplacement(X, Y)
        return -2 / self.h * contract('mn, mnj -> mnj', self.kx(X, Y), displacements)

    def hesskx(self, X, Y):
        displacements = getPairwiseDisplacement(X, Y)
        a = -2 / self.h * contract('mne, mni -> mnie', self.gkx(X, Y), displacements)
        b = -2 / self.h * contract('mn, ie -> mnie', self.kx(X, Y), np.eye(X.shape[1]))
        return a + b

def getGaussianKernelWithDerivatives_identity(X, h, get_hesskx = False):
    kx = np.exp(-getPairwiseDistance(X, X, metric='sqeuclidean') / h)
    displacements = getPairwiseDisplacement(X, X)
    gkx = -2 / h * contract('mn, mnj -> mnj', kx, displacements)
    if get_hesskx is True:
        hesskx =  -2 / h * (contract('mne, mni -> mnie', gkx, displacements) +
                            contract('mn, ie -> mnie', kx, np.eye(X.shape[1])))
        return kx, gkx, hesskx
    else:
        return kx, gkx

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
class IMQ_identity():
    def __init__(self, beta=-0.5, c=1):
        assert(beta < 0)
        assert(c > 0)
        self.beta = beta
        self.c = c
    def k(self, x, y):
        tmp1 = self.c ** 2 + np.linalg.norm(x - y) ** 2
        return tmp1 ** self.beta
    def gk(self, x, y):
        tmp1 = self.c ** 2 + np.linalg.norm(x - y) ** 2
        return 2 * self.beta * tmp1 ** (self.beta - 1) * (x - y)
    def hessk(self, x, y):
        tmp1 = self.c ** 2 + np.linalg.norm(x - y) ** 2
        a = 4 * self.beta * (self.beta - 1) * tmp1 ** (self.beta - 2) * np.outer(x - y, x - y)
        b = 2 * self.beta * tmp1 ** (self.beta - 1) * np.eye(x.size)
        return a + b
    def kx(self, X, Y):
        pairwise_distance_squared = getPairwiseDistance(X, Y, metric='sqeuclidean')
        tmp1 = (self.c ** 2 + pairwise_distance_squared)
        return tmp1 ** self.beta
    def gkx(self, X, Y):
        pairwise_distance_squared = getPairwiseDistance(X, Y, metric='sqeuclidean')
        pairwise_displacement = getPairwiseDisplacement(X, Y)
        tmp1 = (self.c ** 2 + pairwise_distance_squared)
        return contract('mn, mni -> mni', 2 * self.beta * tmp1 ** (self.beta - 1), pairwise_displacement)
    def hesskx(self, X, Y):
        pairwise_distance_squared = getPairwiseDistance(X, Y, metric='sqeuclidean')
        pairwise_displacement = getPairwiseDisplacement(X, Y)
        tmp1 = (self.c ** 2 + pairwise_distance_squared)
        a = 4 * self.beta * (self.beta - 1) * contract('mn, mnj, mni -> mnij', tmp1 ** (self.beta-2), pairwise_displacement, pairwise_displacement)
        b = 2 * self.beta * contract('mn, ij -> mnij', tmp1 ** (self.beta - 1), np.eye(X.shape[1]))
        return a + b

def getIMQ_kernelWithDerivatives_identity(X, c, beta, get_hesskx=False):
    pairwise_distance_squared = getPairwiseDistance(X, X, metric='sqeuclidean')
    pairwise_displacement = getPairwiseDisplacement(X, X)
    tmp1 = (c ** 2 + pairwise_distance_squared)
    kx = tmp1 ** beta
    gkx = contract('mn, mni -> mni', 2 * beta * tmp1 ** (beta - 1), pairwise_displacement)
    if get_hesskx is True:
        a = 4 * beta * (beta - 1) * contract('mn, mnj, mni -> mnij', tmp1 ** (beta-2), pairwise_displacement, pairwise_displacement)
        b = 2 * beta * contract('mn, ij -> mnij', tmp1 ** (beta - 1), np.eye(X.shape[1]))
        hesskx = a + b
        return kx, gkx, hesskx
    else:
        return kx, gkx

class Gaussian_metric():
    def __init__(self, h=1, M=None):
        """
        Args:
            h (float): Kernel bandwidth
            M (array): D x D square, symmetric, positive definite metric.
        """
        assert(h > 0)
        self.h = h
        if M is not None:
            assert np.allclose(M, M.T) # Ensure metric is symmetric
            try:
                np.linalg.cholesky(M)
                self.M = M
            except:
                raise('Metric is not positive.')

    #####################################################
    # Evaluate at a particle
    #####################################################
    def k(self, x, y):
        quad_form = np.dot(x-y, self.M @ (x-y)) / self.h
        return np.exp(-quad_form)

    def gk(self, x, y):
        return -2 / self.h * self.k(x,y) * self.M @ (x-y)

    def hessk(self, x, y):
        return -2 / self.h * (np.outer(self.gk(x,y), self.M @ (x-y)) + self.k(x, y) * self.M)

    ######################################################
    # Evaluate on an ensemble of particles
    ######################################################
    def kx(self, X, Y):
        pairwise_displacements = getPairwiseDisplacement(X, Y)
        pairwise_distances_squared = contract('mni, ij, mnj -> mn', pairwise_displacements, self.M, pairwise_displacements)
        return np.exp(-pairwise_distances_squared / self.h)

    def gkx(self, X, Y):
        kx = self.kx(X, Y)
        pairwise_displacement = getPairwiseDisplacement(X, Y)
        return -2 / self.h * contract('mn, ej, mnj -> mne', kx, self.M, pairwise_displacement)

    def hesskx(self, X, Y):
        kx = self.kx(X, Y)
        gkx = self.gkx(X, Y)
        pairwise_displacement = getPairwiseDisplacement(X, Y)
        tmp1 = contract('mne, ij, mnj -> mnie', gkx, self.M, pairwise_displacement)
        tmp2 = contract('ie, mn -> mnie', self.M, kx)
        return -2 / self.h * (tmp1 + tmp2)

def getGaussianKernelWithDerivatives_metric(X, M=None, h=1, get_hesskx=False, get_tr_hesskx=False):
    if M is None:
        M = np.eye(X.shape[1])
    pairwise_displacements = getPairwiseDisplacement(X, X)
    M_pairwise_displacements = contract('ij, mnj -> mni', M, pairwise_displacements)
    kx = np.exp(-contract('mni, mni -> mn', pairwise_displacements, M_pairwise_displacements) / h)
    gkx = -2 / h * contract('mn, mni -> mni', kx, M_pairwise_displacements)
    if get_hesskx is True:
        a = contract('mne, mni -> mnie', gkx, M_pairwise_displacements)
        b = contract('mn, ie -> mnie', kx, M)
        hesskx = -2/h * (a + b)
        return kx, gkx, hesskx
    elif get_tr_hesskx is True:
        a = contract('mni, mni -> mn', gkx, M_pairwise_displacements)
        b = contract('mn, ii -> mn', kx, M)
        hesskx = -2/h * (a + b)
        return kx, gkx, hesskx
    else:
        return kx, gkx

# def getDeltaM_h_WithDerivatives(X, Y, M, h):

class IMQ_metric():
    def __init__(self, M, h, c=1, beta=-0.5):
        self.M = M
        self.h = h
        self.c = c
        self.beta = beta
        self.D = lambda x, y: np.dot(x-y, self.M @ (x - y)) / self.h
        self.gD = lambda x, y: 2 * self.M @ (x - y) / self.h
        self.hessD = 2 * self.M / h # constant

    def k(self, x, y):
        tmp1 = (self.c ** 2 + self.D(x,y))
        return tmp1 ** self.beta
    def gk(self, x, y):
        tmp1 = (self.c ** 2 + self.D(x,y))
        return self.beta * tmp1 ** (self.beta - 1) * self.gD(x, y)
    def hessk(self, x, y):
        tmp1 = (self.c ** 2 + self.D(x,y))
        gd = self.gD(x,y)
        return self.beta * (self.beta - 1) * tmp1 ** (self.beta - 2) * np.outer(gd, gd) \
                + self.beta * tmp1 ** (self.beta - 1) * self.hessD

    def kx(self, X, Y):
        pairwise_displacements = getPairwiseDisplacement(X, Y)
        M_pairwise_displacements = contract('ij, mnj -> mni', self.M, pairwise_displacements)
        D = contract('mni, mni -> mn', pairwise_displacements, M_pairwise_displacements) / self.h
        tmp1 = (self.c ** 2 + D)
        return tmp1 ** self.beta

    def gkx(self, X, Y):
        pairwise_displacements = getPairwiseDisplacement(X, Y)
        M_pairwise_displacements = contract('ij, mnj -> mni', self.M, pairwise_displacements)
        D = contract('mni, mni -> mn', pairwise_displacements, M_pairwise_displacements) / self.h
        tmp1 = (self.c ** 2 + D)
        gD = 2 * M_pairwise_displacements / self.h
        return contract('mn, mne -> mne', self.beta * tmp1 ** (self.beta - 1), gD)

    def hesskx(self, X, Y):
        pairwise_displacements = getPairwiseDisplacement(X, Y)
        M_pairwise_displacements = contract('ij, mnj -> mni', self.M, pairwise_displacements)
        D = contract('mni, mni -> mn', pairwise_displacements, M_pairwise_displacements) / self.h
        gD = 2 * M_pairwise_displacements / self.h
        tmp1 = (self.c ** 2 + D)
        hessD = (2 * self.M / self.h)
        tmp2 = contract('mne, mnf -> mnef', gD, gD)
        return contract('mn, mnef -> mnef', self.beta * (self.beta - 1) * tmp1 ** (self.beta - 2), tmp2)   \
                + contract('mn, ef -> mnef', self.beta * tmp1 ** (self.beta - 1), hessD)


def getIMQ_metricWithDerivatives(X, M=None, h=1, beta=-0.5, c=1, get_hesskx=False, get_tr_hesskx=False):
    if M is None:
        M = np.eye(X.shape[1])
    pairwise_displacements = getPairwiseDisplacement(X, X)
    M_pairwise_displacements = contract('ij, mnj -> mni', M, pairwise_displacements)
    D = contract('mni, mni -> mn', pairwise_displacements, M_pairwise_displacements) / h
    gD = 2 * M_pairwise_displacements / h
    tmp1 = (c ** 2 + D)
    hessD = (2 * M / h)
    tmp2 = contract('mne, mnf -> mnef', gD, gD)

    kx = tmp1 ** beta
    gkx = contract('mn, mne -> mne', beta * tmp1 ** (beta - 1), gD)
    if get_hesskx is True:
        hesskx = contract('mn, mnef -> mnef', beta * (beta - 1) * tmp1 ** (beta - 2), tmp2) \
               + contract('mn, ef -> mnef', beta * tmp1 ** (beta - 1), hessD)
        return kx, gkx, hesskx
    if get_tr_hesskx is True:
        tr_hesskx = contract('mn, mnff -> mn', beta * (beta - 1) * tmp1 ** (beta - 2), tmp2) \
                 + contract('mn, ff -> mn', beta * tmp1 ** (beta - 1), hessD)
        return kx, gkx, tr_hesskx
    return kx, gkx

class linear():
    def __init__(self):
        pass
    def k(self, x, y):
        return 1 + np.dot(x,y)
    def gk(self, x, y):
        return y
    def hessk(self, x, y):
        return np.zeros((x.size, x.size))
    def kx(self, X, Y):
        return 1 + contract('mi, ni -> mn', X, Y)
    def gkx(self, X, Y):
        return contract('m, ne -> mne', np.ones(X.shape[0]), Y)
    def hesskx(self, X, Y):
        N = X.shape[0]
        D = X.shape[1]
        return np.zeros((N, N, D, D))

class linear_metric():
    def __init__(self, Q):
        self.Q = Q
        pass
    def k(self, x, y):
        return np.dot(x, self.Q @ y) + 1
    def gk1(self, x, y):
        return self.Q @ y
    def gk2(self, x, y):
        return self.Q @ x
    def hessk11(self, x, y):
        return np.zeros((x.size, x.size))
    def hessk12(self, x, y):
        return self.Q
    def kx(self, X, Y):
        return contract('mi, ij, nj -> mn', X, self.Q, Y) + 1
    def gkx1(self, X, Y):
        return contract('m, ej, nj -> mne', np.ones(X.shape[0]), self.Q, Y)
    # def gkx2(self, X, Y):

    def hesskx(self, X, Y):
        N = X.shape[0]
        D = X.shape[1]
        return np.zeros((N, N, D, D))

################################################################
# Linear kernel with metric preconditioning
################################################################
class class_linear_metric():
    def __init__(self, M):
        self.M = M
    def k(self, x, y):
        return np.dot(x, self.M @ y) + 1

def get_linear_metric(X, M=None,
                      get_kx=True,
                      get_gkx1=False,
                      get_gkx2=False,
                      get_tr_hesskx11=False,
                      get_hesskx11=False,
                      get_hesskx12=False):
    N = X.shape[0]
    D = X.shape[1]
    if M is None:
        M = np.eye(D)
    tmp1 = contract('ij, nj -> ni', M, X)
    results = []
    if get_kx is True:
        kx = contract('mi, ni -> mn', X, tmp1) + 1
        results.append(kx)
    if get_gkx1 is True:
        gkx1 = contract('m, ne -> mne', np.ones(N), tmp1)
        results.append(gkx1)
    if get_gkx2 is True:
        if get_gkx1 is True:
            gkx2 = contract('mne -> nme', gkx1)
        else:
            gkx2 = contract('m, ne -> nme', np.ones(N), tmp1)
        results.append(gkx2)
    if get_tr_hesskx11 is True:
        hesskx11 = np.zeros((N, N))
        results.append(hesskx11)
    if get_hesskx11 is True:
        hesskx11 = np.zeros((N, N, D, D))
        results.append(hesskx11)
    if get_hesskx12 is True:
        hesskx12 = contract('mn, ef -> mnef', np.ones((N, N)), M)
        results.append(hesskx12)
    return tuple(results)

################################################################
# IMQ kernel with metric preconditioning
################################################################
class class_IMQ_metric():
    def __init__(self,
                 M=None,
                 h=None,
                 beta=-0.5,
                 c=1):
        self.beta = beta
        self.c = c
        self.M = M
        if h is None:
            self.h = 1
        else:
            self.h = h
    def k(self, x, y):
        if self.M is None:
            self.M = np.eye(x.size)
        tmp1 = self.c ** 2 + np.dot(x - y, self.M @ (x - y))
        return tmp1 ** self.beta

def get_IMQ_metric(X,
                   beta=-0.5,
                   c=1,
                   M=None,
                   h=1,
                   get_kx=True,
                   get_gkx1=False,
                   get_gkx2=False,
                   get_tr_hesskx11=False,
                   get_hesskx11=False,
                   get_hesskx12=False):
    if M is None:
        M = np.eye(X.shape[1])
    results = []
    Delta = getPairwiseDisplacement(X, X)
    tmp1 = contract('ij, mnj -> mni', M, Delta) # Needed to get the derivatives of D(x, y)
    tmp2 = c ** 2 + contract('mni, mni -> mn', Delta, tmp1) / h
    if get_kx is True:
        kx = tmp2 ** beta
        results.append(kx)
    if get_gkx1 is True:
        gD = 2 * tmp1 / h
        gkx1 = contract('mn, mne -> mne', beta * tmp2 ** (beta - 1), gD)
        results.append(gkx1)
    if get_gkx2 is True:
        if get_gkx1 is True:
            gkx2 = -1 * gkx1
        else:
            gD = 2 * tmp1 / h
            gkx2 = contract('mn, mne -> mne', beta * tmp2 ** (beta - 1), -1 * gD)
        results.append(gkx2)
    if get_tr_hesskx11 is True:
        tr_hesskx11 = contract('mn, mne, mne -> mn', beta * (beta - 1) * tmp2 ** (beta - 2), gD, gD) \
                 + contract('mn, ee -> mn', beta * tmp2 ** (beta - 1), 2 * M / h)
        results.append(tr_hesskx11)
    if get_hesskx11 is True:
        if get_gkx1 is False and get_gkx2 is False:
            gD = 2 * tmp1 / h
        hesskx11 = contract('mn, mne, mnf -> mnef', beta * (beta - 1) * tmp2 ** (beta - 2), gD, gD) \
                 + contract('mn, ef -> mnef', beta * tmp2 ** (beta - 1), 2 * M / h)
        results.append(hesskx11)
    if get_hesskx12 is True:
        if get_hesskx11 is True:
            hesskx12 = -1 * hesskx11
        else:
            hesskx12 = - contract('mn, mne, mnf -> mnef', beta * (beta - 1) * tmp2 ** (beta - 2), gD, gD) \
                       - contract('mn, ef -> mnef', beta * tmp2 ** (beta - 1), 2 * M / h)
        results.append(hesskx12)
    return tuple(results)

################################################################
# Random feature RBF kernel with metric preconditioning
################################################################
class class_random_feature_RBF_metric():
    def __init__(self, M, D, l=5, h=2, w=None, v=None):
        np.random.seed(1)
        if w is None:
            self.w = np.random.multivariate_normal(np.zeros(D), np.eye(D), l)
        else:
            self.w = w
        if v is None:
            self.v = np.random.uniform(0, 2 * np.pi)
        else:
            self.v = v
        self.h = h
        self.l = l
        self.sqrtM = np.linalg.cholesky(M) # Lower triangular
    def k(self, x, y):
        phix = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, self.sqrtM @ x) / self.h + self.v)
        phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, self.sqrtM @ y) / self.h + self.v)
        return contract('l, l -> ', phix, phiy) / self.l

def get_randomRBF_metric(X, w=None, v=None, M=None, h=2, l=5,
                         get_kx=True,
                         get_gkx1=False,
                         get_gkx2=False,
                         get_tr_hesskx11=False,
                         get_hesskx11=False,
                         get_hesskx12=False):
    results = []
    D = X.shape[1]
    # np.random.seed(1)
    if M is None:
        M = np.eye(D)
    if w is None:
        w = np.random.multivariate_normal(np.zeros(D), np.eye(D), l)
    else:
        w = w
    if v is None:
        v = np.random.uniform(0, 2 * np.pi)
    else:
        v = v
    sqrtM = np.linalg.cholesky(M)
    A = contract('li, ij -> lj', w, sqrtM) / h
    tmp2 = contract('lj, mj -> ml', A, X) + v
    phi = np.sqrt(2) * np.cos(tmp2)
    if get_kx is True:
        kx = contract('ml, nl -> mn', phi, phi) / l
        results.append(kx)
    if get_gkx1 is True:
        gphi = contract('ml, le -> mle', -np.sqrt(2) * np.sin(tmp2), A)
        gkx1 = contract('mle, nl -> mne', gphi, phi) / l
        results.append(gkx1)
    if get_gkx2 is True:
        if get_gkx1 is True:
            gkx2 = contract('mne -> nme', gkx1)
        else:
            gphi = contract('ml, le -> mle', -np.sqrt(2) * np.sin(tmp2), A) # Copy
            gkx2 = contract('mle, nl -> nme', gphi, phi) / l
        results.append(gkx2)
    if get_tr_hesskx11 is True:
        hessphi11 = contract('ml, le, lf -> mlef', -phi, A, A)
        tr_hesskx11 = contract('mlee, nl -> mn', hessphi11, phi) / l
        results.append(tr_hesskx11)
    if get_hesskx11 is True:
        if get_tr_hesskx11 is False:
            hessphi11 = contract('ml, le, lf -> mlef', -phi, A, A)
        hesskx11 = contract('mlef, nl -> mnef', hessphi11, phi) / l
        results.append(hesskx11)
    if get_hesskx12 is True:
        if get_gkx1 is False and get_gkx2 is False:
            gphi = contract('ml, le -> mle', -np.sqrt(2) * np.sin(tmp2), A) # Copy
        hesskx12 = contract('mle, nlf -> mnef', gphi, gphi) / l
        results.append(hesskx12)
    return tuple(results)

################################################################
# Pointwise preconditioned RBF
################################################################
class class_pointwise_preconditioned_RBF():
    def __init__(self, Ml, X, h1=2, h2=2):
        self.h1 = h1 # Bandwidth
        self.h2 = h2 # weight extent
        self.Ml = Ml # Pointwise preconditioning
        self.X = X # Anchor points
    def k(self, x, y): # X represents the ND anchor points
        D = x.shape[0]
        z = (self.h2 * np.pi) ** (D / 2) * np.linalg.det(self.Ml) ** (-0.5)
        # w(x)
        deltas_anchorx = contract('li, lij, lj -> l', x-self.X, self.Ml, x-self.X) #  agreed
        nx = np.exp(-1/self.h2 * deltas_anchorx) / z # agreed
        wx = nx / np.sum(nx)
        # w(y)
        deltas_anchory = contract('li, lij, lj -> l', y-self.X, self.Ml, y-self.X)
        ny = np.exp(-1/self.h2 * deltas_anchory) / z
        wy = ny / np.sum(ny)
        # Calculate kernel
        deltas = contract('i, lij, j -> l', x-y, self.Ml, x-y)
        k = np.exp(-1/self.h1 * deltas)
        return np.sum(k * wx * wy)

def get_pointwise_preconditioned_RBF(X, Ml, h1=2, h2=2,
                                     get_kx=True,
                                     get_gkx1=False,
                                     get_gkx2=False,
                                     get_tr_hesskx11=False,
                                     get_hesskx11=False,
                                     get_hesskx12=False):
    results = []
    D = X.shape[1]
    N = X.shape[0]
    # z = (2 * np.pi) ** (D / 2) * np.linalg.det(Ml) ** (-0.5)
    z = (h2 * np.pi) ** (D / 2) * np.linalg.det(Ml) ** (-0.5)
    Delta = getPairwiseDisplacement(X, X)
    # norm = contract('mni, lij, mnj -> lmn', Delta, Ml, Delta)
    Ml_Delta = contract('lij, mnj -> lmni', Ml, Delta)
    norm = contract('mni, lmni -> lmn', Delta, Ml_Delta)
    n = contract('lml, l -> ml', np.exp(-1/h2 * norm), 1 / z)
    # n = contract('lml, l -> ml', np.exp(-0.5 * norm), 1 / z)
    tmp1 = 1 / np.sum(n, axis=1)
    w = contract('ml, m -> lm', n, tmp1) # This step is correct
    kx_tmp = np.exp(-norm / h1)
    kx = contract('lmn, lm, ln -> mn', kx_tmp, w, w)
    results.append(kx)

    gkx_tmp = -2 / h1 * contract('lmn, lmni -> lmni', kx_tmp, Ml_Delta) # agreed

    # n_e = -contract('ml, lmle -> mle', n, Ml_Delta) # agreed
    n_e = -2/h2 * contract('ml, lmle -> mle', n, Ml_Delta) # agreed

    tmp2 = np.sum(n_e, axis=1) # agreed
    w_mle = contract('mle, m -> mle', n_e, tmp1) - \
          contract('ml, m, me -> mle', n, tmp1 ** 2, tmp2) # agreed

    gkx1 = contract('lmne, lm, ln -> mne', gkx_tmp, w, w) + \
          contract('lmn, mle, ln -> mne', kx_tmp, w_mle, w)

    results.append(gkx1)
    # gkx2 = None
    # hesskx12 = None
    # results.append(gkx2)
    # results.append(hesskx12)
    return tuple(results)

def get_pointwise_preconditioned_RBF_v2(X, Ml, h=2,
                                     get_kx=True,
                                     get_gkx1=False,
                                     get_gkx2=False,
                                     get_tr_hesskx11=False,
                                     get_hesskx11=False,
                                     get_hesskx12=False):
    results = []
    D = X.shape[1]
    N = X.shape[0]
    # z = (h2 * np.pi) ** (D / 2) * np.linalg.det(Ml) ** (-0.5)
    z = (h * np.pi) ** (D / 2) * np.linalg.det(Ml) ** (-0.5)
    Delta = getPairwiseDisplacement(X, X)
    Ml_Delta = contract('lij, mnj -> lmni', Ml, Delta)
    norm = contract('mni, lmni -> lmn', Delta, Ml_Delta)
    # n = contract('lml, l -> ml', np.exp(-1/h2 * norm), 1 / z)
    tmp0 = np.exp(-1/h * norm)
    n = contract('lml, l -> ml', tmp0, 1 / z)
    # n = contract('lml, l -> ml', np.exp(-0.5 * norm), 1 / z)
    tmp1 = 1 / np.sum(n, axis=1)
    w = contract('ml, m -> lm', n, tmp1) # This step is correct
    # kx_tmp = np.exp(-norm / h1)
    kx = contract('lmn, lm, ln -> mn', tmp0, w, w)
    results.append(kx)

    gkx_tmp = -2 / h * contract('lmn, lmni -> lmni', tmp0, Ml_Delta) # agreed

    # n_e = -contract('ml, lmle -> mle', n, Ml_Delta) # agreed
    n_e = -2/h * contract('ml, lmle -> mle', n, Ml_Delta) # agreed

    tmp2 = np.sum(n_e, axis=1) # agreed
    w_mle = contract('mle, m -> mle', n_e, tmp1) - \
            contract('ml, m, me -> mle', n, tmp1 ** 2, tmp2) # agreed

    gkx1 = contract('lmne, lm, ln -> mne', gkx_tmp, w, w) + \
           contract('lmn, mle, ln -> mne', tmp0, w_mle, w)

    results.append(gkx1)
    # gkx2 = None
    # hesskx12 = None
    # results.append(gkx2)
    # results.append(hesskx12)
    return tuple(results)







# def getPointwisePreconditionedKernel(l, w):
#     h2 = 1
#     kx = np.zeros((self.nParticles, self.nParticles))
#     gkx1 = np.zeros((self.nParticles, self.nParticles, self.DoF))
#     for m in self.nParticles:
#         contract('mni, -> ml')
#         gauss = lambda X: np.exp(1/h2 ))
#         kx += kx_l * gauss(m, m)





class random():
    def __init__(self, h, l, D):
        self.h = h
        self.l = l
        self.D = D
        self.w = np.random.multivariate_normal(np.zeros(self.D), np.eye(self.D), self.l)
        self.v = np.random.uniform(0, 2 * np.pi)
    def k(self, x, y):
        phix = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, x) / self.h + self.v)
        phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, y) / self.h + self.v)
        return contract('l, l -> ', phix, phiy) / self.l
    def gk(self, x, y):
        phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, y) / self.h + self.v)
        tmp1 = np.sin(contract('li, i -> l', self.w, x) / self.h + self.v)
        gphix = -np.sqrt(2) * contract('l, le -> le', tmp1, self.w / self.h)
        return contract('le, l -> e', gphix, phiy) / self.l
    def hessk(self, x, y):
        phix = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, x) / self.h + self.v)
        phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, y) / self.h + self.v)
        tmp1 = contract('l, le, lf -> lef', -phix / self.h ** 2, self.w, self.w)
        return contract('lef, l -> ef', tmp1, phiy) / self.l
    def kx(self, X, Y):
        tmp1 = contract('li, mi -> ml', self.w, X)
        tmp2 = contract('li, mi -> ml', self.w, Y)
        phix = np.sqrt(2) * np.cos(tmp1 / self.h + self.v)
        phiy = np.sqrt(2) * np.cos(tmp2 / self.h + self.v)
        return contract('ml, nl -> mn', phix, phiy) / self.l
    def gkx(self, X, Y):
        tmp1 = contract('li, mi -> ml', self.w, X)
        tmp2 = contract('li, mi -> ml', self.w, Y)
        # phix = np.sqrt(2) * np.cos(tmp1 / self.h + self.v)
        phiy = np.sqrt(2) * np.cos(tmp2 / self.h + self.v)
        tmp3 = -np.sqrt(2) * np.sin(tmp1 / self.h + self.v)
        gphix = contract('ml, le -> mle', tmp3, self.w / self.h)
        return contract('mle, nl -> mne', gphix, phiy) / self.l
    def hesskx(self, X, Y):
        tmp1 = contract('li, mi -> ml', self.w, X)
        tmp2 = contract('li, mi -> ml', self.w, Y)
        phix = np.sqrt(2) * np.cos(tmp1 / self.h + self.v)
        phiy = np.sqrt(2) * np.cos(tmp2 / self.h + self.v)
        hessphix = contract('ml, le, lf -> mlef', -phix / self.h ** 2, self.w, self.w)
        return contract('mlef, nl -> mnef', hessphix, phiy) / self.l

# class random_metric():
#     def __init__(self, h, l, D, Q):
#         self.h = h
#         self.l = l
#         self.D = D
#         self.w = np.random.multivariate_normal(np.zeros(self.D), np.eye(self.D), self.l)
#         self.v = np.random.uniform(0, 2 * np.pi)
#         self.Q = Q
#         self.sqrtQ = scipy.linalg.sqrtm(self.Q)
#     def k(self, x, y):
#         phix = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, self.sqrtQ @ x) / self.h + self.v)
#         phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, y) / self.h + self.v)
#         return contract('l, l -> ', phix, phiy) / self.l
#     def gk(self, x, y):
#         phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, y) / self.h + self.v)
#         tmp1 = np.sin(contract('li, i -> l', self.w, x) / self.h + self.v)
#         gphix = -np.sqrt(2) * contract('l, le -> le', tmp1, self.w / self.h)
#         return contract('le, l -> e', gphix, phiy) / self.l
#     def hessk(self, x, y):
#         phix = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, x) / self.h + self.v)
#         phiy = np.sqrt(2) * np.cos(contract('li, i -> l', self.w, y) / self.h + self.v)
#         tmp1 = contract('l, le, lf -> lef', -phix / self.h ** 2, self.w, self.w)
#         return contract('lef, l -> ef', tmp1, phiy) / self.l


class fourierRandomFeatureGaussian:
    def __init__(self, L, DoF):
        """
        Random fourier feature representation of radial basis function given in Detommasso
        k(x, y) = exp{(x-y).T M (x-y) / h}
        Args:
            L (int): Number of features to use
            DoF (int): Dimension
        """
        self.L = L
        self.DoF = DoF
        self.W = np.random.normal(0, 1, size=(self.L, self.DoF))
        self.b = np.random.uniform(0, 2 * np.pi, self.L)

    def _k(self, x, y, h, M):
        k = 0
        U = scipy.linalg.cholesky(M)
        x_ = np.sqrt(2 * h) * U @ x
        y_ = np.sqrt(2 * h) * U @ y
        for l in range(self.L):
            phix_l = np.sqrt(2) * np.cos(np.dot(self.W[l], x_) / h + self.b[l])
            phiy_l = np.sqrt(2) * np.cos(np.dot(self.W[l], y_) / h + self.b[l])
            k += phix_l * phiy_l
        return k / self.L

    def getKernelWithDerivatives(self, X, h, M, get_gkx1=True):
        """
        Get kernel and derivative efficiently
        Args:
            X (array): N x DoF array of particles
            h (float): Kernel bandwidth
            M (array): DoF x DoF metric
            get_gkx1 (boolean): True to return kernel and gradient as tuple

        Returns: (tuple)

        """
        # Transform Detommasso kernel to Liu kernel
        # U = scipy.linalg.sqrtm(M)
        U = scipy.linalg.cholesky(M)
        X_ = np.sqrt(2 * h) * contract('ij, mj -> mi', U, X)
        # Calculate argument of trig function
        arg = contract('li, mi -> lm', self.W, X_) / h + self.b[..., np.newaxis]
        phi = np.sqrt(2) * np.cos(arg)
        kx = contract('lm, ln -> mn', phi, phi) / self.L
        if get_gkx1 is not True:
            return kx
        WU = contract('li, ie -> le', self.W, U)
        gphi = -2 / np.sqrt(h) * contract('lm, lj -> lmj', np.sin(arg), WU)
        gkx1 = contract('lmi, ln -> mni', gphi, phi) / self.L
        return (kx, gkx1)


class fourierRandomFeatureGaussian_scaled:
    def __init__(self, L, DoF):
        """
        Implements random feature kernel eq(6) from "Stein Variational Gradient Descent as Moment Matching"
        (Liu 2018) with constant preconditioning as suggested in Eq(12) of "Stein Variational Gradient Descent
        with Matrix-Valued Kernels" (Wang 2019) using averaged Gauss-Newton preconditioner suggested in Eq(19)
        of "A Stein Variational Newton Method"

        Args:
            L (int): Number of features to use
            DoF (int): Dimension
        """
        self.L = L
        self.DoF = DoF
        self.W = np.random.normal(0, 1, size=(self.L, self.DoF))
        self.b = np.random.uniform(0, 2 * np.pi, self.L)
        # self.b = np.random.uniform(0, 2 * np.pi)

    def _k(self, x, y, h, M):
        k = 0
        U = scipy.linalg.cholesky(M)
        x_ = U @ x
        y_ = U @ y
        for l in range(self.L):
            phix_l = np.sqrt(2) * np.cos(np.dot(self.W[l], x_) / h + self.b[l])
            phiy_l = np.sqrt(2) * np.cos(np.dot(self.W[l], y_) / h + self.b[l])
            k += phix_l * phiy_l
        return k / self.L

    def getKernelWithDerivatives(self, X, h, M, get_gkx1=True):
        """
        Get kernel and derivative efficiently
        Args:
            X (array): N x DoF array of particles
            h (float): Kernel bandwidth
            M (array): DoF x DoF metric
            get_gkx1 (boolean): True to return kernel and gradient as tuple

        Returns: (tuple)

        """
        # Transform Detommasso kernel to Liu kernel
        # U = scipy.linalg.cholesky(M)
        U = scipy.linalg.sqrtm(M)
        X_ = contract('ij, mj -> mi', U, X)
        # Calculate argument of trig function
        arg = contract('li, mi -> lm', self.W, X_) / h + self.b[..., np.newaxis]
        # arg = contract('li, mi -> lm', self.W, X_) / h + self.b
        phi = np.sqrt(2) * np.cos(arg)
        kx = contract('lm, ln -> mn', phi, phi) / self.L
        if get_gkx1 is not True:
            return kx
        WU = contract('li, ie -> le', self.W, U)
        gphi = -np.sqrt(2) / h * contract('lm, lj -> lmj', np.sin(arg), WU)
        gkx1 = contract('lmi, ln -> mni', gphi, phi) / self.L
        return (kx, gkx1)






    # # D = contract('mni, mni -> mn', delta, M_delta) / h
    # # tmp1 = (c ** 2 + D)
    # if get_kx is True:
    #     kx = tmp1 ** beta
    #     results.append(kx)
    # if get_gkx1 is True:
    #
    #
    #
    #
    #
    #
    # hessD = (2 * M / h)
    # tmp2 = contract('mne, mnf -> mnef', gD, gD)
    # gkx = contract('mn, mne -> mne', beta * tmp1 ** (beta - 1), gD)
    #
    # if get_hesskx is True:
    #     hesskx = contract('mn, mnef -> mnef', beta * (beta - 1) * tmp1 ** (beta - 2), tmp2) \
    #              + contract('mn, ef -> mnef', beta * tmp1 ** (beta - 1), hessD)
    #     return kx, gkx, hesskx
    # if get_tr_hesskx is True:
    #     tr_hesskx = contract('mn, mnff -> mn', beta * (beta - 1) * tmp1 ** (beta - 2), tmp2) \
    #                 + contract('mn, ff -> mn', beta * tmp1 ** (beta - 1), hessD)
    #     return kx, gkx, tr_hesskx

########################################################
# Evaluate efficiently on a single ensemble
#########################################################
# def getGaussianKernelWithDerivatives(X, h, M, get_hesskx=False):
#     pairwise_displacements = getPairwiseDisplacement(X, X)
#     M_pairwise_displacements = contract('ij, mnj -> mni', M, pairwise_displacements)
#     kx = np.exp(-contract('mni, mni -> mn', pairwise_displacements, M_pairwise_displacements) / h)
#     gkx = -2 / h * contract('mn, mni -> mni', kx, M_pairwise_displacements)
#     if get_hesskx is True:
#         hesskx = -2 / h * (contract('mne, mni -> mnie', gkx, M_pairwise_displacements)
#                            + contract('mn, ie -> mnie', kx, M))
#         return kx, gkx, hesskx
#     else:
#         return kx, gkx

def main():
    pass
if __name__ is '__main__':
    main()