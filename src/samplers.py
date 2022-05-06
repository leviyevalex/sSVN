import logging
from tqdm import trange  # Use to add a progress bar to loop
import copy
import h5py
import os, sys
import logging.config
from pyinstrument import Profiler
import numpy as np
from opt_einsum import contract
from time import time, sleep
import scipy
from scipy import sparse, linalg, spatial
import scipy.sparse.linalg

log = logging.getLogger(__name__)
# log.addHandler(logging.StreamHandler(stream=sys.stdout))
np.seterr(over='raise')
np.seterr(invalid='raise')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class samplers:
    def __init__(self, model, nIterations, nParticles, profile=None):
        # Quick settings checks
        assert(nParticles > 0)
        assert(nIterations > 0)
        # Setup file paths, logger, and profiler.
        now = str(int(time()))
        sleep(1) # Output path depends on 'now'. Make sure new instances are spaced out enough to be unique.
        log.debug('DEBUG: ROOT path %s' % ROOT_DIR)
        self.OUTPUT_DIR = os.path.join(ROOT_DIR, 'outdir')
        self.RUN_OUTPUT_DIR = os.path.join(self.OUTPUT_DIR, now)
        self.history_path = os.path.join(self.RUN_OUTPUT_DIR, 'output_data.h5')
        # Create folders that organize results
        if os.path.isdir(self.OUTPUT_DIR) == False:
            os.mkdir(self.OUTPUT_DIR)
        if os.path.isdir(self.RUN_OUTPUT_DIR) == False:
            os.mkdir(self.RUN_OUTPUT_DIR)
        # Logger setup
        LOG_OUTPUT_DIR = os.path.join(self.RUN_OUTPUT_DIR, 'info.log')
        fh = logging.FileHandler(filename=LOG_OUTPUT_DIR, mode='w')
        fh.setLevel(logging.INFO)
        log.addHandler(fh)

        if profile == True or profile == 'True' or profile == "'True'":
            self.profile = True
        else:
            self.profile = False
        self.model = model
        self.nParticles = nParticles
        self.nIterations = nIterations
        self.DoF = model.DoF
        self.dim = self.DoF * self.nParticles

    def apply(self, method='SVGD', eps=0.1):
        """
        Evolves a set of particles according to (method) with step-size (eps).
        Args:
            method (str): Sampler method chosen.
            Options available: 'SVGD', 'sSVGD', 'BDSVN', 'SVN', 'sSVN'
        Returns (dict):
        'path_to_results' : Path to output h5 file with particle history, etc...
        'outdir_path': Path to output directory

        """
        if self.profile == True:
            profiler = Profiler()
            profiler.start()
        # np.random.seed(int(time())) # Enable for randomness
        np.random.seed(1) # Enable for reproducibility
        try:
            X = self.model._newDrawFromPrior(self.nParticles) # Initial set of particles
            h = 2 * self.DoF
            with trange(self.nIterations) as ITER:
                for iter_ in ITER:
                    if method == 'SVGD':
                        gmlpt = self.model.getGradientMinusLogPosterior_ensemble(X)
                        GN_Hmlpt = self.model.getGNHessianMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0)
                        # M = None
                        kx, gkx1 = self._getKernelWithDerivatives(X, h=h, M=M)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        update = v_svgd * eps
                    elif method == 'sSVGD':
                        gmlpt = self.model.getGradientMinusLogPosterior_ensemble(X)
                        GN_Hmlpt = self.model.getGNHessianMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0)
                        # M = None
                        kx, gkx1 = self._getKernelWithDerivatives(X, h=h, M=M)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        alpha, L_kx = self._getMinimumPerturbationCholesky(kx)
                        if alpha != 0:
                            kx += alpha * np.eye(self.nParticles)
                        v_stc = self._getSVGD_v_stc(L_kx)
                        update = v_svgd * eps + v_stc * np.sqrt(eps)
                    elif method == 'BDSVN':
                        gmlpt = self.model.getGradientMinusLogPosterior_ensemble(X)
                        GN_Hmlpt = self.model.getGNHessianMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0)
                        # M = None
                        kx, gkx1 = self._getKernelWithDerivatives(X, h=h, M=M)
                        import tensorflow as tf
                        # solve_method = 'Cholesky'
                        solve_method = 'CG'
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        HBD = self._getSteinHessianBlockDiagonal(GN_Hmlpt, kx, gkx1)
                        if solve_method == 'Cholesky':
                            LHBD = tf.linalg.cholesky(HBD)
                            v_svn = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis]))
                        elif solve_method == 'CG':
                            cg_maxiter = 10
                            HBDop = tf.linalg.LinearOperatorFullMatrix(HBD, is_self_adjoint=True, is_positive_definite=True)
                            v_svn = tf.linalg.experimental.conjugate_gradient(HBDop, tf.constant(v_svgd), max_iter=cg_maxiter).x.numpy()
                        update = v_svn * eps
                    elif method == 'SVN':
                        gmlpt = self.model.getGradientMinusLogPosterior_ensemble(X)
                        GN_Hmlpt = self.model.getGNHessianMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0)
                        # M = None
                        kx, gkx1 = self._getKernelWithDerivatives(X, h=h, M=M)
                        solve_method = 'CG'
                        # solve_method = 'Cholesky'
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        H = self._getSteinHessianPosdef(GN_Hmlpt, kx, gkx1)
                        if solve_method == 'CG':
                            cg_maxiter = 50
                            alphas = scipy.sparse.linalg.cg(H, v_svgd.flatten(), maxiter=cg_maxiter)[0].reshape(self.nParticles, self.DoF)
                            v_svn = contract('xd, xn -> nd', alphas, kx)
                        elif solve_method == 'Cholesky':
                            lamb = 0.01
                            H = H1 + NK * lamb
                            UH = scipy.linalg.cholesky(H)
                            v_svn = self._getSVN_direction(kx, v_svgd, UH)
                        update = v_svn * eps
                    elif method == 'sSVN':
                        gmlpt = self.model.getGradientMinusLogPosterior_ensemble(X)
                        GN_Hmlpt = self.model.getGNHessianMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0)
                        # M = None
                        kx, gkx1 = self._getKernelWithDerivatives(X, h, M)
                        NK = self._reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx, np.eye(self.DoF)))
                        H1 = self._getSteinHessianPosdef(GN_Hmlpt, kx, gkx1)
                        lamb = 0.01
                        H = H1 + NK * lamb
                        UH = scipy.linalg.cholesky(H)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        v_svn = self._getSVN_direction(kx, v_svgd, UH)
                        v_stc = self._getSVN_v_stc(kx, UH)
                        update = (v_svn) * eps + v_stc * np.sqrt(eps)

                    # Update progress bar
                    # ITER.set_description('Stepsize %f | Median bandwidth: %f | SVN norm: %f | Noise norm: %f | SVGD norm %f | Dampening %f' % (eps, self._bandwidth_MED(X), np.linalg.norm(v_svn), np.linalg.norm(v_stc), np.linalg.norm(v_svgd),  lamb))
                    ITER.set_description('Stepsize %f | Median bandwidth: %f' % (eps, self._bandwidth_MED(X)))

                    # Store relevant per iteration information
                    with h5py.File(self.history_path, 'a') as f:
                        g = f.create_group('%i' % iter_)
                        g.create_dataset('X', data=copy.deepcopy(X))
                        g.create_dataset('h', data=copy.deepcopy(h))
                        g.create_dataset('eps', data=copy.deepcopy(eps))
                        g.create_dataset('gmlpt', data=copy.deepcopy(gmlpt))
                        g.create_dataset('id', data=copy.deepcopy(self.model.id))

                    # Update particles
                    X += update

                # Dynamics completed: Storing data
                with h5py.File(self.history_path, 'a') as f:
                    g = f.create_group('metadata')
                    g.create_dataset('X', data=copy.deepcopy(X))
                    g.create_dataset('nLikelihoodEvals', data=copy.deepcopy(self.model.nLikelihoodEvaluations))
                    g.create_dataset('nGradLikelihoodEvals', data=copy.deepcopy(self.model.nGradLikelihoodEvaluations))
                    g.create_dataset('nParticles', data=copy.deepcopy(self.nParticles))
                    g.create_dataset('DoF', data=copy.deepcopy(self.DoF))
                    g.create_dataset('L', data=copy.deepcopy(iter_ + 1))
                    g.create_dataset('method', data=method)
                    g1 = f.create_group('final_updated_particles')
                    g1.create_dataset('X', data=X)

            # Save profiling results
            if self.profile == True:
                log.info('OUTPUT: Saving profile as html')
                profiler.stop()
                try:
                    with open(os.path.join(self.RUN_OUTPUT_DIR, 'output.html'), "w") as f:
                        profile_html = profiler.output_html()
                        log.info(type(profile_html))
                        f.write(profile_html)
                        log.info('OUTPUT: Successfully saved profile to html.')
                        # log.info(profiler.output_text(unicode=True, color=True))
                except:
                    log.error('OUTPUT: Failed to save profile to html. Trying utf-8', exc_info=True)
                    try:
                        with open(os.path.join(self.RUN_OUTPUT_DIR, 'output.html'), "wb") as f:
                            f.write(profile_html.encode('utf-8'))
                            log.info('OUTPUT: Successfully saved profile to html.')
                    except:
                        log.error('OUTPUT: Failed to save utf-8 profile to html.', exc_info=True)

            log.info('OUTPUT: Run completed successfully! Data stored in:\n %s' % self.history_path)

        except Exception:
            log.error("Error occurred in apply()", exc_info=True)

########################################################################################################################

    # Direction methods (either to make the main code looks clean or so we may reuse it elsewhere)

    def _getSVGD_direction(self, kx, gkx, gmlpt):
        """
        Get SVGD velocity field
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$
            gmlpt (array): N x D array, gradient of minus log target

        Returns:

        """
        v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt) / self.nParticles + np.mean(gkx, axis=0)
        return v_svgd

    def _getSVGD_v_stc(self, L_kx, Bdn=None):
        """
        Get noise injection velocity field for SVGD
        Args:
            L_kx (array): N x N array, lower triangular Cholesky factor of kernel gram matrix $k$
            Bdn (array): D x N array, standard normal noise

        Returns:

        """
        if Bdn is None:
            Bdn = np.random.normal(0, 1, (self.DoF, self.nParticles))
        return np.sqrt(2 / self.nParticles) * contract('mn, in -> im', L_kx, Bdn).flatten(order='F').reshape(self.nParticles, self.DoF)

    def _getSVN_direction(self, kx, v_svgd, UH):
        """
        Get SVN velocity field
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            v_svgd (array): N x D array, SVGD velocity field
            UH (array): ND x ND array, upper triangular Cholesky factor of SVN Hessian $H$

        Returns: (array) N x D array, SVN direction

        """
        alphas = scipy.linalg.cho_solve((UH, False), v_svgd.flatten()).reshape(self.nParticles, self.DoF)
        v_svn = contract('mn, ni -> mi', kx, alphas)
        return v_svn

    def _getSVN_v_stc(self, kx, UH):
        """
        Get noise injection velocity field for SVN
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            UH (array): ND x ND array, upper triangular Cholesky factor of SVN Hessian $H$

        Returns: (array) N x D array, noise injection for SVN

        """
        B = np.random.normal(0, 1, self.dim)
        tmp1 = scipy.linalg.solve_triangular(UH, B, lower=False).reshape(self.nParticles, self.DoF)
        return np.sqrt(2 / self.nParticles) * contract('mn, ni -> mi', kx, tmp1)

    def _getSteinHessianBlockDiagonal(self, Hmlpt, kx, gkx):
        """
        Calculate block diagonal SVN Hessian $H_{BD}$ by taking block diagonal of both $H_1, H_2$
        Args:
            Hmlpt (array): N x D x D array, Hessian target evaluated at particle n $- \nabla^2 \ln \pi(z_n)$
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$

        Returns: (array) N x D x D block diagonal SVN Hessian $H_{BD}$

        """
        return (contract('mn, nij -> mij' , kx ** 2, Hmlpt) + contract('mni, mnj -> mij', gkx, gkx)) / self.nParticles

    def _getSteinHessianPosdef(self, Hmlpt, kx, gkx):
        """
        Calculate SVN Hessian $H = H_1 + H_2$.
        Note: If H_1 is made positive-definite (with Gauss-Newton approximation for example),
        then adding H_2 block diagonal ensures H is positive definite.
        Args:
            Hmlpt (array): N x D x D array, Hessian target evaluated at particle n $- \nabla^2 \ln \pi(z_n)$
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$

        Returns: (array) ND x ND SVN Hessian $H$

        """
        H1 = contract("xy, xz, xbd -> yzbd", kx, kx, Hmlpt)
        H2 = contract('xzi, xzj -> zij', gkx, gkx) # Only calculate block diagonal
        H1[range(self.nParticles), range(self.nParticles)] += H2
        return self._reshapeNNDDtoNDND(H1 / self.nParticles)

    def _getMinimumPerturbationCholesky(self, x, jitter=1e-9):
        """
        Solution adapted from TensorFlow GitHub page
        Args:
            x (array): "nearly" positive definite matrix
            jitter (float): How much to add to x $x + jitter * I$ where $I$ is the identity matrix.

        Returns: (float, array) Required jitter to produce decomposition and corresponding lower triangular Cholesky factor

        """
        try:
            cholesky = np.linalg.cholesky(x)
            return 0, cholesky
        except Exception:
            while jitter < 1.0:
                try:
                    cholesky = np.linalg.cholesky(x + jitter * np.eye(x.shape[0]))
                    log.warning('CHOLESKY: Matrix not positive-definite. Adding alpha = %.2E' % jitter)
                    return jitter, cholesky
                except Exception:
                    jitter = jitter * 10
            raise Exception('CHOLESKY: Factorization failed.')

    def _hyperbolic_schedule(self, t, T, c=1.3, p=5):
        """
        Hyperbolic annealing schedule
        Args:
            t (int): Current iteration
            T (int): Total number of iterations
            c (float): Controls where transition begins
            p (float): Exponent determining speed of transition between phases

        Returns: (float)

        """
        return np.tanh((c * t / T) ** p)

    def _cyclic_schedule(self, t, T, p=5, C=5):
        """
        Cyclic annealing schedule
        Args:
            t (int): Current iteration
            T (int): Total number of iterations
            p (float): Exponent determining speed of transition between phases
            C (int): Number of cycles

        Returns:

        """
        tmp = T / C
        return (np.mod(t, tmp) / tmp) ** p

    # Stochastic SVN : Reshaping methods
    def _reshapeNNDDtoNDND(self, H):
        """

        Args:
            H (array): Takes N x N x d x d tensor to reshape

        Returns (array): Rearranged H into an ND x ND tensor with N blocks of size d x d along the diagonal

        """
        return H.swapaxes(1, 2).reshape(self.dim, self.dim)

    def _reshapeNDNDtoNNDD(self, H):
        """
        # Adapted from TensorFlow library
        Undoes action of self._reshapeNNDDtoNDND
        Args:
            H (array): Takes Nd x Nd tensor with N blocks of size d x d along the diagonal

        Returns (array): Returns N x N x d x d tensor of blocks

        """
        block_shape = np.array((self.DoF, self.DoF))
        new_shape = tuple(H.shape // block_shape) + tuple(block_shape)
        new_strides = tuple(H.strides * block_shape) + H.strides
        return np.lib.stride_tricks.as_strided(H, shape=new_shape, strides=new_strides)

    # Bandwidth selection
    def _bandwidth_MED(self, X):
        """
        Median bandwidth selection method
        Args:
            X (array): N x D array of particle positions

        Returns: (float) kernel bandwidth $h$

        """
        pairwise_distance = scipy.spatial.distance_matrix(X, X)
        median = np.median(np.trim_zeros(pairwise_distance.flatten()))
        return median ** 2 / np.log(self.nParticles + 1)

    def _getKernelWithDerivatives(self, X, h, M=None):
        """
        Computes radial basis function (Gaussian) kernel with optional "metric" - See (Detommasso 2018)
        Args:
            X (array): N x d array of particles
            h (float): Kernel bandwidth
            M (array): d x d positive semi-definite metric.

        Returns (tuple): N x N kernel gram matrix, N x N x d gradient of kernel (with respect to first slot of kernel)

        """

        displacement_tensor = self._getPairwiseDisplacement(X)
        if M is not None:
            U = scipy.linalg.cholesky(M)
            X = contract('ij, nj -> ni', U, X)
            displacement_tensor = contract('ej, mnj -> mne', M, displacement_tensor)
        kx = np.exp(-scipy.spatial.distance_matrix(X, X) ** 2 / h)
        gkx1 = -2 * contract('mn, mne -> mne', kx, displacement_tensor) / h
        # test_gkx = -2 / h * contract('mn, ie, mni -> mne', kx, U, displacement_tensor)
        return kx, gkx1

    # Useful distance functions

    def _getPairwiseDisplacement(self, X):
        return X[:,np.newaxis,:] - X[np.newaxis,:,:]
