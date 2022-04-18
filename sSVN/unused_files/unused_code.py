# taylor f2 model

def reflective_boundary_exp_individual(self, theta):
    """
    Modifies cost function to include exponential boundary near end of particle support
    Args:
        theta: Single particle position (d x 1 dimensional array)

    Returns: Barrier value at particle location (scalar)

    """
    begin_barrier = 1  # Where curvature of barrier begins
    A = np.abs(1)  # Amplitude on barrier
    curvature = 5  # Controls steepness of barrier
    barrier_function = lambda theta: A * (
                np.exp(-curvature * (theta[0] - begin_barrier)) + np.exp(-curvature * (theta[1] - begin_barrier)))
    return barrier_function(theta)


def reflective_boundary_exp_ensemble(self, thetas):
    """
     Modifies cost function to include exponential boundary near end of particle support
     Evaluates for multiple particles (ensemble)
     Note: This will also work for any number of particles, not necessarily only the whole ensemble.
    Args:
        thetas: position of ensemble (d x n dimensional array)

    Returns: Barrier value at all locations given in thetas (n dim array)

    """
    barrier_function_evaluations = np.apply_along_axis(self.reflective_boundary_exp_individual, 0, thetas)
    return barrier_function_evaluations


def grad_reflective_boundary_exp_individual(self, theta):
    """
    Helper function to get gradient of barrier at particle location
    Args:
        theta: Particle location (d x 1 dimensional array)

    Returns: Gradient at theta (d dimensional array)

    """
    return nd.Gradient(self.reflective_boundary_exp_individual, step=self.gradientStep, method=self.gradientMethod)(
        theta)


def grad_reflective_boundary_exp_ensemble(self, thetas):
    """
    Helper function to get gradient of barrier at all input locations
    Args:
        thetas: position of ensemble (d x n dimensional array)

    Returns: Gradient at all input locations (d x n dimensional array)

    """
    return np.apply_along_axis(self.grad_reflective_boundary_exp_individual, 0, thetas)


def hess_reflective_boundary_exp_individual(self, theta):
    """
    Helper function to get Hessian of barrier at individual particle location
    Args:
        theta: position of particle (d x 1 dimensional array)

    Returns: Hessian evaluated at input particle position (d x d x 1 dimensional array)

    """
    return nd.Hessian(self.reflective_boundary_exp_individual, method=self.hessianMethod, step=self.hessianStep)(theta)


def hess_reflective_boundary_exp_ensemble(self, thetas):
    """
    Helper function to get Hessian of barrier at all particle locations
    Args:
        thetas: position of ensemble (d x n dimensional array)

    Returns: Hessian evaluated at all input particle positions (d x d x n dimensional array)

    """
    return np.apply_along_axis(self.hess_reflective_boundary_exp_individual, 0, thetas)



#  SVI

    def hess_reflective_boundary_exp_ensemble(self, thetas):
        """
        Helper function to get Hessian of barrier at all particle locations
        Args:
            thetas: position of ensemble (d x n dimensional array)

        Returns: Hessian evaluated at all input particle positions (d x d x n dimensional array)

        """
        return np.apply_along_axis(self.hess_reflective_boundary_exp_individual, 0, thetas)


    def hess_reflective_boundary_exp_individual(self, theta):
        """
        Helper function to get Hessian of barrier at individual particle location
        Args:
            theta: position of particle (d x 1 dimensional array)

        Returns: Hessian evaluated at input particle position (d x d x 1 dimensional array)

        """
        return nd.Hessian(self.reflective_boundary_exp_individual, method=self.hessianMethod, step=self.hessianStep)(theta)

    def reflective_boundary_exp_individual(self, theta):
        """
        Modifies cost function to include exponential boundary near end of particle support
        Args:
            theta: Single particle position (d x 1 dimensional array)

        Returns: Barrier value at particle location (scalar)

        """
        begin_barrier = 1 # Where curvature of barrier begins
        A = np.abs(1) # Amplitude on barrier
        curvature = 5 # Controls steepness of barrier
        barrier_function = lambda theta: A * (np.exp(-curvature * (theta[0] - begin_barrier)) + np.exp(-curvature * (theta[1] - begin_barrier)))
        return barrier_function(theta)

    def reflective_boundary_exp_ensemble(self, thetas):
        """
         Modifies cost function to include exponential boundary near end of particle support
         Evaluates for multiple particles (ensemble)
         Note: This will also work for any number of particles, not necessarily only the whole ensemble.
        Args:
            thetas: position of ensemble (d x n dimensional array)

        Returns: Barrier value at all locations given in thetas (n dim array)

        """
        barrier_function_evaluations = np.apply_along_axis(self.reflective_boundary_exp_individual, 0, thetas)
        return barrier_function_evaluations

    def grad_reflective_boundary_exp_individual(self, theta):
        """
        Helper function to get gradient of barrier at particle location
        Args:
            theta: Particle location (d x 1 dimensional array)

        Returns: Gradient at theta (d dimensional array)

        """
        return nd.Gradient(self.reflective_boundary_exp_individual, step=self.gradientStep, method=self.gradientMethod)(theta)

    def grad_reflective_boundary_exp_ensemble(self, thetas):
        """
        Helper function to get gradient of barrier at all input locations
        Args:
            thetas: position of ensemble (d x n dimensional array)

        Returns: Gradient at all input locations (d x n dimensional array)

        """
        return np.apply_along_axis(self.grad_reflective_boundary_exp_individual, 0, thetas)


# %%
class BILBY_TAYLORF2():
    def __init__(self, *arg):
        """
        Creates an object to input into SVN class.
        Args:
            *arg ():
            arg1 inject: [m1 > m2] necessary by convention (arg1 is optional)
            arg2 injectSigma: Gaussian std over inject variables for prior (optional)
        """
        self.modelType = 'taylorf2'
        self.randomSeed = 88170235  # Original from bilby tutorial. Seed used to create a unique likelihood
        self.inject = arg[0] if len(arg) > 0 else np.array([2, 2])
        self.injectSigma = arg[1] if len(arg) > 1 else np.array([.2, .2])
        self.injection_parameters = dict(
            mass_1=self.inject[0],
            mass_2=self.inject[1],
            chi_1=0.02,
            chi_2=0.02,
            luminosity_distance=120.,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
            lambda_1=400,
            lambda_2=450,
            iota=.1)
        self.params_copy = copy.deepcopy(self.injection_parameters) # used in the getWaveform method
        self.theta_0 = copy.deepcopy(self.injection_parameters)  # Will be used in likelihood evaluation
        self.DoF = self.inject.size
        self.injection_parameter_order = self.setInjectionParameterOrder()
        self.likelihood = self.create_bilby_likelihood()
        self.m1GaussianPrior = self.setGaussianPrior(self.inject[0], self.injectSigma[0])
        self.m2GaussianPrior = self.setGaussianPrior(self.inject[1], self.injectSigma[1])
        self.gradientStep = 1e-5  # Step size for central differencing the gradient
        self.gradientMethod = 'central'
        self.hessianStep = 1e-5  # Step size for central differencing the hessian
        self.hessianMethod = 'central'
        self.outOfBounds = 0  # What to do if particles pushed out of bounds. 0 = Return zero, 1 = Raise(ValueError), np.inf makes cost function infinite outside valid range
        self.scale_likelihood = 1  # Scalar used to rescale geometry of likelihood
        self.scale_prior = 1  # Scalar used to rescale geometry of prior
        self.logLikelihoodPeak = self.getMinusLogLikelihood_individual(self.inject)
        self.logPriorPeak = self.getMinusLogPrior_individual(self.inject)
        self.include_boundary = False # Includes term in posterior evaluation that will create a boundary around bad inputs
        self.forwardJacobian = nd.Jacobian(self.getWaveform, step=self.gradientStep, method=self.gradientMethod)

    def getWaveform(self, theta):
        """
        Get the gravitational waveform model given parameters

        theta: numpy array of parameters (m1, m2)
        """
        np.random.seed(self.randomSeed)

        # --------------------------------------------------
        # Set duration and sampling frequency for data segment injection.
        # TaylorF2 waveform cuts the signal close to the isco frequency
        # --------------------------------------------------

        duration = 8
        sampling_frequency = 2 * 1570.
        start_time = self.injection_parameters['geocent_time'] + 2 - duration

        waveform_arguments = dict(waveform_approximant='TaylorF2',
                                  reference_frequency=50., minimum_frequency=40.0)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments, start_time = start_time)

        self.params_copy['mass_1'] = theta[0]
        self.params_copy['mass_2'] = theta[1]

        waveform_array = waveform_generator.frequency_domain_strain(parameters = self.params_copy)

        return np.real(waveform_array['plus'])

    def setInjectionParameterOrder(self):
        injection_parameter_order = [
            'mass_1',
            'mass_2',
            'chi_1',
            'chi_2',
            'luminosity_distance',
            'theta_jn',
            'psi',
            'phase',
            'geocent_time',
            'ra',
            'dec',
            'lambda_1',
            'lambda_2',
            'iota'
        ]
        return injection_parameter_order

    def create_bilby_likelihood(self):
        """
        Initializes a TAYLORF2 model using the Bilby inference library
        Returns: a Bilby likelihood object

        """
        np.random.seed(self.randomSeed)

        # --------------------------------------------------
        # Set duration and sampling frequency for data segment injection.
        # TaylorF2 waveform cuts the signal close to the isco frequency
        # --------------------------------------------------

        duration = 8
        sampling_frequency = 2 * 1570.
        start_time = self.injection_parameters['geocent_time'] + 2 - duration

        waveform_arguments = dict(waveform_approximant='TaylorF2',
                                  reference_frequency=50., minimum_frequency=40.0)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments, start_time = start_time)

        # --------------------------------------------------
        # Interferometer Setup
        # --------------------------------------------------

        # Default at design sensitivity and @ 40 Hz.
        # interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
        interferometers = bilby.gw.detector.InterferometerList(['L1'])
        for interferometer in interferometers:
            interferometer.minimum_frequency = 40

        # Data in interferometer is currently just noise governed by the PSD
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
            start_time=start_time)

        interferometers.inject_signal(parameters=self.injection_parameters,
                                      waveform_generator=waveform_generator)

        bilby_likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=interferometers, waveform_generator=waveform_generator,
            time_marginalization=False, phase_marginalization=False,
            distance_marginalization=False)

        return bilby_likelihood

    def getMinusLogLikelihood_individual(self, theta):
        """
        Evaluates the likelihood at a particular point in parameter space
        The previous dictionary copy is just to make code pretty.
        Args:
            theta (d dimensional np.array): Where to evaluate the likelihood in parameter space
        Returns: scalar

        """
        # # only over m1 and m2 for now!
        #
        if theta[0] <= 0 or theta[1] <= 0:
            if self.outOfBounds == 1:
                message = f"\n Input to logLikelihoodEvaluation outside of domain \n mass_1 = {theta[0]} or mass_2 = {theta[1]} <= 0 "
                raise ValueError(message)
            elif self.outOfBounds == 0:
                return 0
            elif self.outOfBounds == np.inf:
                return np.inf

        for pos, var in enumerate(self.injection_parameter_order):
            if pos < self.DoF:
                self.theta_0[var] = theta[pos]
            else:
                break

        self.likelihood.parameters = self.theta_0

        return -1 * self.likelihood.log_likelihood() / self.scale_likelihood

    def getMinusLogLikelihood_ensemble(self, thetas):
        """
        Takes a matrix containing all particle position info and returns an array
        of respective negative log likelihoods a la bilby
        Args:
            self ():
            thetas (d x n ndarray): d = number of parameters, n = number of particles.
            *arg ():

        Returns: n dimensional array of likelihoods corresponding to each particle position

        """

        return np.apply_along_axis(self.getMinusLogLikelihood_individual, 0, thetas)

    def setGaussianPrior(self, theta_i, sigma_i):
        """
        Helper function that sets a gaussian prior using the bilby library
        Args:
            theta_i: mean of the gaussian distribution (float)
            sigma_i: standard deviation (float)

        Returns: bilby gaussian

        """

        return bilby.core.prior.Gaussian(theta_i, sigma_i)

    def getMinusLogPrior_individual(self, theta):
        """
        Helper function to get priors. Assume parameters are independent and distributed via gaussian
        Args:
            theta (): d dimensional array representing position in parameter space

        Returns: scalar.

        """

        if theta[0] <= 0 or theta[1] <= 0:
            if self.outOfBounds == 1:
                message = f"\n Input to getLogPrior outside of domain \n mass_1 = {theta[0]} or mass_2 = {theta[1]} <= 0 "
                raise ValueError(message)
            elif self.outOfBounds == 0:
                pass
                return 0
            elif self.outOfBounds == np.inf:
                return np.inf

        return -1 * (self.m1GaussianPrior.ln_prob(theta[0]) + self.m2GaussianPrior.ln_prob(theta[1]))/self.scale_prior

    def getMinusLogPrior_ensemble(self, thetas):
        """
        Takes a matrix containing all particle position info and returns an array
            of respective negative log priors a la bilby
        Args:
            self ():
            thetas (d x n ndarray): d = number of parameters, n = number of particles.

        Returns: n dimensional array of priors evaluated at each particle position

        """

        return np.apply_along_axis(self.getMinusLogPrior_individual, 0, thetas)


    def getMinusLogPosterior_individual(self, theta):
        """
        Evaluates posterior at single particle location
        Args:
            theta: Particle location (d x 1 dimensional array)

        Returns: Posterior evaluated at theta (scalar)

        """
        if self.include_boundary == True:
            return self.getMinusLogLikelihood_individual(theta) + self.getMinusLogPrior_individual(theta) \
                   + self.reflective_boundary_exp_individual(theta)
        elif self.include_boundary == False:
            return self.getMinusLogLikelihood_individual(theta) + self.getMinusLogPrior_individual(theta)


    def getGradientMinusLogPrior_individual(self, theta):
        """
        Helper function to get gradient of minus log prior for individual particle
        Args:
            theta: particle location (d x 1 dimensional array)

        Returns: gradient evaluated at particle position (d dimensional array)

        """
        return nd.Gradient(self.getMinusLogPrior_individual, step=self.gradientStep, method=self.gradientMethod)(theta)

    def getGradientMinusLogPrior_ensemble(self, thetas):
        """
        Method to get gradient of the negative log prior
        Args:
            thetas (d x n np.array): array containing all particle positions

        Returns: d x n np.array of gradients at all particle positions

        """

        return np.apply_along_axis(self.getGradientMinusLogPrior_individual, 0, thetas)



    def buildHessianPosterior(self, thetas, nParticles, DoF):
        """
        Stores the
        Args:
            thetas:
            nParticles:
            DoF:

        Returns:

        """


    def getGradientMinusLogLikelihood_ensemble(self, thetas):
        """"
        Method to get gradient of the negative log likelihood
        Args:
           thetas (d x n np.array): array containing all particle positions

        Returns: d x n np.array of gradients at all particle positions

        """
        return np.apply_along_axis(self.getGradientMinusLogLikelihood_individual, 0, thetas)

    def getGradientMinusLogPosterior_individual(self, theta):
        """
        Helper method to get the gradient evaluated at a particular particle
        Args:
            theta: particle position (d x 1 dimensional array)

        Returns: Gradient evaluated at a particle (d dimensional array)

        """
        return nd.Gradient(self.getMinusLogPosterior_individual, step=self.gradientStep, method=self.gradientMethod)(theta)
        # return self.getGradientMinusLogLikelihood_1_part(theta) + self.getGradientMinusLogPrior_1_part(theta) + self.get_grad_reflections_1_part(theta)

    def getGradientMinusLogPosterior_ensemble(self, thetas):
        """
        Method to get gradient of negative log posterior
        Args:
            thetas: array of all particle positions (d x n array)

        Returns: array of gradients evaluated at each particle position (d x n array)

        """
        # return self.getGradientMinusLogPrior(thetas) + self.getGradientMinusLogLikelihood(thetas)
        return np.apply_along_axis(self.getGradientMinusLogPosterior_individual, 0, thetas)

    def getGradientMinusLogLikelihood_individual(self, theta):
        """
        Helper function to evaluate gradient of minus log likelihood at a particle
        Args:
            theta: particle position (d x 1 dimensional array)

        Returns: gradient of minus log likelihood evaluated at a particle (d dimensional array)

        """
        return nd.Gradient(self.getMinusLogLikelihood_individual, step=self.gradientStep, method=self.gradientMethod)(theta)

    def getHessianMinusLogPrior_individual(self,theta):
        """
          Method to get hessian of negative log prior
          Args:
              theta: particle position (d x 1 array)

          Returns: hessian evaluated at each particle position (d x d x 1 array)

          """

        return nd.Hessian(self.getMinusLogPrior_individual, method=self.hessianMethod, step=self.hessianStep)(theta)

    def getHessianMinusLogPrior_ensemble(self, thetas):
        """
        Method to get hessian of negative log prior
        Args:
            thetas: array of all particle positions (d x n array)

        Returns: array of hessians evaluated at each particle position (d x d x n array)

        """

        return np.apply_along_axis(self.getHessianMinusLogPrior_individual,0, thetas)

    def getHessianMinusLogLikelihood_individual(self,theta):
        """
        Helper method to get hessian of minus log likelihood for individual particle
        Args:
            theta: particle position (n x 1 dimensional array)

        Returns: hessian of minus log likelihood evaluated at particle position

        """
        return nd.Hessian(self.getMinusLogLikelihood_individual, method=self.hessianMethod, step=self.hessianStep)(theta)

    def getHessianMinusLogPosterior_individual(self, theta):
        """
        Helper method to get hessian of minus log posterior at particle
        Args:
            theta: particle position (d x 1 dimensional array)

        Returns: hessian at minus log posterior (d x d dimensional array)

        """

        return nd.Hessian(self.getMinusLogPosterior_individual, method=self.hessianMethod, step=self.hessianStep)(theta)

    def getHessianMinusLogLikelihood_ensemble(self, thetas):
        """
        Method to get hessian of negative log likelihood
        Args:
            thetas: array of all particle positions (d x n array)

        Returns: array of hessians evaluated at each particle position (d x d x n array)

        """
        return np.apply_along_axis(self.getHessianMinusLogLikelihood_individual, 0, thetas)


    def getGaussNewtonHessianMinusLogPosterior_ensemble (self, thetas):
        return np.apply_along_axis(self.getGaussNewtonHessianMinusLogPosterior_individual, 0, thetas)

    def getGaussNewtonHessianMinusLogPosterior_individual(self, theta):
        jac = self.forwardJacobian(theta)
        return jac.T @ jac


# Algorithm side
# Needed to compute first variation
# gmlpt = self.getGradientMinusLogPosterior_ensemble(self.particles)
# Defined near Eq (18) [Stein variational newton]
# if optimizer == 'SVN':
# Hmlpt = self.model.getGaussNewtonHessianMinusLogPosterior_ensemble(self.particles)

# Positive definite approximation of Hmlpt
# JTJ = self.grad_outer_products(gmlpt)
# Hmlpt = JTJ


# M = np.mean(JTJ, 2)
# M = np.mean(Hmlpt, 2)
# M = np.identity(self.DoF)
# lmbda, v = np.linalg.eig(M)


# Enforcing Positive definiteness of metric via eigenvalue decomposition
# self.metric = np.abs(lmbda) * v @ v.T


# self.gmlpt = gmlpt

# Hmlpt_posdef = np.zeros_like(Hmlpt)

# Make the Hessian's at every particle position positive definite
# counter1 = 0
# for counter1 in range(self.nParticles):
#     lmbda1, v1 = np.linalg.eig(Hmlpt[:, :, counter1])
#     Hmlpt_posdef[:, :, num] = np.abs(lmbda1) * v1 @ v1.T

# Positive definite version of Hmlpt
# Hmlpt = Hmlpt_posdef
#####################################################
#####################################################
#####################################################
# self.buildHessianKernel()
# self.buildGradUpdate()
# self.createLineSearchCrossSection()
#####################################################
#####################################################
#####################################################

# Old Code
# for i_ in range(self.nParticles):
#     # Stein flow: Eq (9) Stein variational newton, Eq (8) SVGD
#     mgJ, kerns, gkerns = self.evaluate_mgJ(particle=i_, distribution=self.particles, gmlpt=gmlpt)
#     # print(kerns)
#     # print(gkerns)
#     # Symmetric-form block diagonal approximation: Eq (14) SVN
#     ####################### fixed gkern #################################
#     HJ = np.mean(Hmlpt * kerns ** 2, 2) + 4 * np.matmul(gkerns, gkerns.T) / self.nParticles
#     ####################### ORIGINAL ########################
#     # HJ = np.mean(Hmlpt * kerns ** 2, 2) + np.matmul(gkerns, gkerns.T) / self.nParticles
#


#     # Solving for coefficients Eq (17) SVN
#
#     ###############################################
#     # Get search direction
#     ###############################################
#     try:
#         log.info('Approximate hessian step taken')
#
#         direction = np.linalg.solve(HJ, mgJ)
#         # Q[:, i_] = np.linalg.solve(HJ, mgJ)
#     except:
#         HJ = np.identity(self.DoF)
#         Q[:, i_] = np.linalg.solve(HJ, mgJ)
#         log.info('Gradient step taken')
#     ###############################################
#     ###############################################
#     ###############################################
#
#     HJ = np.identity(self.DoF)
#     Q[:, i_] = np.linalg.solve(HJ, mgJ)


# search_direction_criteria = np.dot(Q[:, i_], -mgJ)

# start = time.time()
# box = np.array([np.linalg.eigvals(HJ)[0], np.linalg.eigvals(HJ)[1], candidate_step])

# large_norm_direction_list = self.return_large_norm_cols(Q)
# Q = self.normalize_cols_w_list(Q, large_norm_direction_list)

# log.info('Following directions are huge, ie > 20 in euclidean norm:')
# huge_search_directions_after_fix = self.return_large_norm_cols(Q)
# log.info(huge_search_directions_after_fix)
# log.info('Total cost function decrease is: %f' % total_cost_function_decrease)

# self.searchDirection = self.normalize_cols(Q)

# self.searchDirection = Q

# self.searchDirection = self.normalize_cols(Q)

# figp = self.plot_linesearch_ensemble(amax, iter = iter_)

# Fixing particles pushed out of bounds issue
# if np.any(self.particles < 0):
#     pass
#     list_revalidate, Hmlpt, gmlpt = self.revalidate_positions(Hmlpt, gmlpt, iter_)
#     log.info('Following particles were pushed out of bounds:')
#     log.info(list_revalidate)
#     raise ValueError('Particles pushed out of support. Ending iterations and creating animation.')
#     RKHS_norm_bfu = np.inf
#     phi_zero_bfu = np.inf

# RKHS_norm_previous = RKHS_norm_bfu
# phi_previous = phi_zero_bfu
# return self.history_path


# build block matrix methods
# ########################
# # For debugging purposes
# ########################
# oproduct_loop = np.zeros((self.DoF, self.DoF))
# gn_loop = np.zeros((self.DoF, self.DoF))
# for l in range(self.nParticles): # For the outer products
#     gn_loop += GN_Hmlpt[:, :, l] * self.k_gram[l, m] * self.k_gram[l, n]
#     oproduct_loop += np.outer(self.gradient_k_gram[l, m, :], self.gradient_k_gram[l, n, :])
#
# final = (gn_loop + oproduct_loop)/self.nParticles
# Vectorized term implementation
# gn_vector = np.einsum('dem, m, m -> de', GN_Hmlpt, self.k_gram[:, m], self.k_gram[:, n])
# oproduct_vector = self.gradient_k_gram[:, m, :].T.dot(self.gradient_k_gram[:, n, :])
# percent_diff = oproduct_vector / gn_vector * 100
# block = (gn_vector + oproduct_vector) / self.nParticles
# print(gn_vector == gn_loop)
# print(oproduct_loop == oproduct_vector)

# next_block = calculateBlockEinsum(GN_Hmlpt, self.nParticles, m, n, self.k_gram, self.gradient_k_gram)
# ########################
# # For debugging purposes
# ########################
# oproduct_loop = np.zeros((self.DoF, self.DoF))
# gn_loop = np.zeros((self.DoF, self.DoF))
# for l in range(self.nParticles): # For the outer products
#     gn_loop += GN_Hmlpt[:, :, l] * self.k_gram[l, m] * self.k_gram[l, n]
#     oproduct_loop += np.outer(self.gradient_k_gram[l, m, :], self.gradient_k_gram[l, n, :])
# Vectorized implementation
# gn_vector = np.einsum('dem, m, m -> de', GN_Hmlpt, self.k_gram[:, m], self.k_gram[:, n])
# oproduct_vector = self.gradient_k_gram[:, m, :].T.dot(self.gradient_k_gram[:, n, :])
# percent_diff = oproduct_vector / gn_vector * 100
# block = (gn_vector + oproduct_vector) / self.nParticles
# print(gn_vector == gn_loop)
# print(oproduct_loop == oproduct_vector)
# #########################
# # End
# #########################
#rosen
        # gradf = numpy.mat([(-2 + 2 * x - 400 * (-x ** 2 + y) * x) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2),
        #                    (-200 * x ** 2 + 200 * y) / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2)])
        #
        # return -1 * np.log((1-x) ** 2 + 100 * (y - x ** 2) ** 2)
        #
        # return  1 / ((1 - x) ** 2 + 100 * (-x ** 2 + y) ** 2) * \
        #        np.array([2 * (1 - x) + 400 * x * (-x ** 2 + y), 200 * (-x ** 2 + y)])

#%%
# import numpy as np
# kern = np.array([[1, 1, 1],
#                  [1, 2, 3],
#                  [4, 5, 6]])
# a = np.array([[1, 2, 3],
#               [1, 2, 3]])
# nParticles = 3
# DoF = 2
#
# w = np.zeros((DoF, nParticles))
# for z in range(nParticles):
#     w_ind = np.zeros(DoF)
#     for n in range(nParticles):
#         w_ind = w_ind + a[:, n] * kern[n, z]
#     w[:,z] = w_ind
#

# cleaning house on algorithm

    # def convergance_checker(self):
    #     RKHS_norm_bfu = self.RKHS_norm(gmlpt)
    #
    #     phi_zero_bfu = self.phi_for_ensemble(0)
    #
    #     log.info('RKHS norm is %.4e, im1 = %.4e' % (RKHS_norm_bfu, RKHS_norm_previous))
    #     log.info('cost is %.10f, im1 = %.10f' % (phi_zero_bfu, phi_previous))
    #
    #     RKHS_norm_increases = RKHS_norm_bfu > RKHS_norm_previous
    #     cost_increases = phi_zero_bfu > phi_previous
    #
    #     if RKHS_norm_increases and cost_increases:
    #         converge_condition_hits += 1
    #         log.info('Update didn\'t decrease RKHS norm and cost. Hits: %i' % converge_condition_hits)
    #         if converge_condition_hits > 1:
    #             # pass # DONT LOOK AT CONVERGENCE
    #             pass
    def stepsize_naive(self):
        #####################################################
        log.info('Searching for stepsize...')
        #####################################################
        try:
            candidate_step = \
                scalar_search_wolfe2(self.phi_for_ensemble, nd.Derivative(self.phi_for_ensemble), maxiter=maxiter, amax=amax)[0]
        except OverflowError:
            log.info(OverflowError)
            pass
        if candidate_step is not None:
            self.stepsize = candidate_step
            log.info('Stepsize found: %f' % self.stepsize)
        else:
            if iter_ != 0:
                log.info('No stepsize found, dividing previous stepsize by 2...')
                self.stepsize /= 2.
                candidate_step = self.stepsize
            else:
                log.info('No stepsize found, setting stepsize = 1 for this case')
                self.stepsize = 1
                candidate_step = self.stepsize

                log.info('Time taken to find stepsize: %.4f, maxiter = %i, step = %f' % (time.time() - start, maxiter, candidate_step))

        #####################################################
        #####################################################
        #####################################################

    def KL_cost(self, positions):
        """
        Helper function to evaluate wolfe conditions in linesearch
        Args:
            positions: particle positions (d x n array)
        Returns: Cost (float)
        """
        cost = np.mean(self.model.getMinusLogPosterior(positions)) / self.nParticles
        return cost

    def grad_KL_cost(self, positions):
        """
        Helper function to evaluate wolfe conditions in line search
        Args:
            positions: particle positions (d x n array)
        Returns: Gradient information at each particle (d x n array)
        """
        gradient = self.model.getGradientMinusLogPosterior(positions)/self.nParticles
        return gradient



    def resetParticles(self, range = None):
        """
        Sets and resets particle positions in parameter space
        Args:
            range: range in parameter space in which to initialize particles for taylorf2 (array)

        Returns: initial particle distribution [dof x n] (array)

        """
        if self.model.modelType == 'gauss':
            self.particles = np.random.normal(scale=1, size=(self.DoF, self.nParticles))
            self.auxilliaryParticles = np.random.normal(scale=1, size=(self.DoF, self.nParticles))

        elif self.model.modelType == 'taylorf2':
            # Truncate gaussian within range
            begin = range[0]
            end = range[1]

            a0, b0 = (begin - self.model.inject[0]) / self.model.injectSigma[0], (end - self.model.inject[0]) / \
                     self.model.injectSigma[0]
            a1, b1 = (begin - self.model.inject[0]) / self.model.injectSigma[0], (end - self.model.inject[0]) / \
                     self.model.injectSigma[0]

            mass1_init = stats.truncnorm(a0, b0, loc=self.model.inject[0], scale=self.model.injectSigma[0]).rvs(
                self.nParticles).T
            mass2_init = stats.truncnorm(a1, b1, loc=self.model.inject[1], scale=self.model.injectSigma[1]).rvs(
                self.nParticles).T
            self.particles = np.vstack((mass1_init, mass2_init))
            self.auxilliaryParticles = np.vstack((mass1_init, mass2_init))
        # elif self.model.modelType == 'taylorf2':
        #     begin = range[0]
        #     end = range[1]
        #     self.particles = np.random.uniform(begin, end, size=[self.model.inject.size, self.nParticles])
    def reset_particle(self, ID = None):
        """
        resets a particles position in parameter space
        Args:
            range: range in parameter space in which to initialize particle for taylorf2 (array)
            ID: particle identification. eg, particle # 23 (int)
        Returns: Nothing. Resets given particle position

        """
        if self.model.modelType == 'gauss':
            self.particles = np.random.normal(scale=1, size=(self.DoF, self.nParticles))
        elif self.model.modelType == 'taylorf2':
            # Truncate gaussian within range
            begin = self.particleRange[0]
            end = self.particleRange[1]

            a0, b0 = (begin - self.model.inject[0]) / self.model.injectSigma[0], (end - self.model.inject[0]) / \
                     self.model.injectSigma[0]
            a1, b1 = (begin - self.model.inject[0]) / self.model.injectSigma[0], (end - self.model.inject[0]) / \
                     self.model.injectSigma[0]

            mass1_init = stats.truncnorm(a0, b0, loc=self.model.inject[0], scale=self.model.injectSigma[0]).rvs(
                1).T
            mass2_init = stats.truncnorm(a1, b1, loc=self.model.inject[1], scale=self.model.injectSigma[1]).rvs(
                1).T

            self.particles[0, ID] = mass1_init
            self.particles[1, ID] = mass2_init

    def valid_position(self, position):
        """
        Helper function to check if position is along valid support
        Args:
            position: Particle position vector ([d x 1] array)

        Returns: True if in support. False if not. (bool)

        """
        if np.any(position < 0):
            return False
        else:
            return True

    def revalidate_positions(self, Hmlpt, gmlpt, iter):
        iterator = np.nditer(self.particles, order='F', flags=['external_loop'])
        # Used to iterate through two numpy arrays columnwise at the same time
        counter = 0
        list = []
        for vector in iterator:
            # negative_definite = np.all(np.linalg.eigvals(Hmlpt[:, :, counter]) < 0) and iter // 5
            if (self.valid_position(vector) != True):
                self.reset_particle(counter)
                list.append(counter)
                Hmlpt[:, :, counter] = self.model.getGNHessianMinusLogPosterior_individual(vector)
                gmlpt[:, counter] = self.model.getGradientMinusLogPosterior_individual(vector)
            counter += 1
        # log.info('Revalidated following particles:')
        # log.info(list)
        for i in list:
            pass
            # log.info('New particle position:')
            # log.info(self.particles[:, i])
        return list, Hmlpt, gmlpt

    def phi_for_particle(self, a, **kwargs):
        """
        Decoupled cost contribution due to single particle. Equation taken from Appendix in pSVN
        (THIS IS FOR A SINGLE TERM IN THE SUM!!!!!)
        Args:
            a: stepsize (float)
            i: particle number (int)
        Returns: Single term evaluated in the sum in the cost function evaluated at given stepsize
        """

        i = self.step_n
        phi_contribution_i = self.model.getMinusLogPosterior_individual(self.particles[:, i] + a * self.searchDirection[:, i])
        return phi_contribution_i / self.nParticles
    # def evaluate_kernel_for_particle(self):
    def phi_line_plot(self, beginning_vector, direction, length_of_line):
        """
        Plot arbitrary line of posterior
        Args:
            beginning_vector: beginning point to define line (d dim vector)
            direction: gives direction to construct line (d dim vector)
            length_of_line: how far to plot
        Returns: Single term evaluated in the sum in the cost function evaluated at given stepsize
        """

        grid = np.arange(0, length_of_line, 0.01)
        posterior_line = lambda a: self.model.getMinusLogPosterior_individual(beginning_vector + a * direction)
        posterior_line_vectorized = np.vectorize(posterior_line)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel('stepsize a')
        ax.set_ylabel('Cost')
        ax.set_title('Begin [%f, %f] Direction [%f, %f]' %
                     (beginning_vector[0], beginning_vector[1],
                      direction[0], direction[1]), fontdict={'fontsize': 20})
        ax.plot(grid, posterior_line_vectorized(grid))

        fig.savefig('line_plot_posterior.png')


def d_phi_for_particle(self, a, **kwargs):
    """
    Derivative of phi with respect to stepsize a.
    Phi is the line on which we're searching. R -> R
    Using Nocedal notation
    Args:
        a: stepsize (float)
        Keyword Args:
        i: particle number (int)
        gmlpt: minus gradient of cost functional minus log posterior given distribution (d x n array)

    Returns: Slope of cost function evaluated at given stepsize
    """
    i = self.step_n
    gmlpt = self.gmlpt
    particle_perturb_positions = np.copy(self.particles)
    particle_perturb_positions[:, i] = particle_perturb_positions[:, i] + a * self.searchDirection[:, i]
    mgJ_perturbed = self.evaluate_mgJ(particle=i, distribution=particle_perturb_positions, gmlpt=gmlpt)[0]
    return np.dot(-mgJ_perturbed, self.searchDirection[:, i])


def phi_for_ensemble(self, a):
    """
    Defines the KL-Divergence cost function from pSVN in terms of phi: R -> R (using Nocedal notation)
    Args:
        a: stepsize (float)
    Returns: Cost function evaluated at given stepsize
    """
    temp = self.model.getMinusLogPosterior_ensemble(self.particles + a * self.searchDirection)
    cost = np.mean(temp)
    return cost


def d_phi_ensemble(self, a):
    """
    Defines the derivative of the KL-Divergence cost function from pSVN. R -> R
    Args:
        a: stepsize (float)
    Returns: Derivative of cost function evaluated at given stepsize
    """
    derivative = np.sum((self.model.getGradientMinusLogPosterior_ensemble(
        self.particles + a * self.searchDirection)) * self.searchDirection) / self.nParticles
    return derivative


def d_phi_ensemble_num(self, a):
    """
    Numerically defines the derivative of the KL-Divergence cost function from pSVN . R -> R
    Args:
        a: stepsize (float)
    Returns: Derivative of cost function evaluated at given stepsize
    """
    derivative = nd.Derivative(self.phi_for_ensemble)
    return derivative(a)


def d_phi_metric(self, a, **kwargs):
    grad = self.model.getGradientMinusLogPosterior(self.particles + a * self.searchDirection)
    metric = self.metric
    iterator = np.nditer((grad, self.searchDirection), order='F', flags=['external_loop'])
    # Used to iterate through two numpy arrays columnwise at the same time
    total = 0
    for grad_col, q_col in iterator:
        total += np.dot(grad_col.T, (metric @ q_col))
    return total / self.nParticles


def d_phi_identity_metric(self, a, **kwargs):
    grad = self.model.getGradientMinusLogPosterior(self.particles + a * self.searchDirection)
    metric = np.identity(self.DoF)
    iterator = np.nditer((grad, self.searchDirection), order='F', flags=['external_loop'])
    # Used to iterate through two numpy arrays columnwise at the same time
    total = 0
    for grad_col, q_col in iterator:
        total += np.dot(grad_col.T, (metric @ q_col))
    return total / self.nParticles


def RKHS_norm(self, dxn):
    """
    computes norm of object in RKHS
    Args:
        dxn: An array of size (d x n) that belongs to the RKHS

    Returns: Sum of gradient norm's that belong to each particle.

    """
    array_of_norms = np.apply_along_axis(np.linalg.norm, 0, dxn)
    gradient_magnitude = np.sum(array_of_norms)
    return gradient_magnitude


def normalize_cols_w_list(self, X, list):
    for i in list:
        X[:, i] = X[:, i] / np.linalg.norm(X[:, i], ord=2, axis=0, keepdims=True)

    return X


def normalize_cols(self, X):
    """
    Normalize columns of a matrix
    Args:
        X: Matrix (array)

    Returns: Matrix with column vectors normalized (array)

    """
    return X / np.linalg.norm(X, ord=2, axis=0, keepdims=True)


def return_large_norm_cols(self, X):
    large_search_direction_list = []
    counter = 0
    for i in np.linalg.norm(X, ord=2, axis=0, keepdims=True):
        if np.linalg.norm(i) > 20:
            large_search_direction_list.append(counter)
        counter += 1
    return large_search_direction_list


def grad_outer_products(self, A):
    iterator = np.nditer(A, order='F')
    # Used to iterate through two numpy arrays columnwise at the same time
    counter = 0

    for i in range(self.nParticles):
        grad = A[:, i]
        block = np.outer(grad, grad)
        if i == 0:
            stack = block
        elif i > 0:
            stack = np.dstack((stack, block))
    return stack


def perturbation_matrix_all(self):
    """
    Creates a perturbation for each particle thats normalized.
    Args:
        list: Which particles to specifically perturb if wanted.

    Returns: A matrix which stores these perturbation values for each particle

    """
    np.random.seed(np.int(time.time()))
    P = np.reshape(np.random.rand(self.DoF * self.nParticles), (self.DoF, self.nParticles))
    return P


def perturbation_matrix(self, list):
    """
    Creates a perturbation for each particle in list (normalized)
    Args:
        list: Which particles to specifically perturb if wanted.

    Returns: A matrix which stores these perturbation values for each particle

    """
    # P = np.reshape(np.random.rand(self.DoF * self.nParticles),(self.DoF, self.nParticles))
    P = np.zeros((self.DoF, self.nParticles))
    for n in list:
        np.random.seed(np.int(time.time()))
        P[:, n] = np.reshape(np.random.rand(self.DoF), (self.DoF))
    return P


def is_pos_def(self, x):
    return np.all(np.linalg.eigvals(x) > 0)


def cholesky_identity_add_individual(self, A):
    beta = np.linalg.norm(A)
    diagonal_elements = np.diag(A)
    max_iter = 1000
    L = None
    if np.min(diagonal_elements) > 0:
        tau = 0
    else:
        tau = beta / 2
    for i in range(max_iter):
        try:
            L = np.linalg.cholesky(A + tau * np.identity(self.DoF))
        except:
            tau = np.max([2 * tau, beta / 2])
        if L is not None:
            return tau
    log.info('Failed to find tau to add for following matrix:')
    print(A)
    raise ValueError('Cholesky tau not found after %i iterations' % max_iter)


def cholesky_modify_add_identity(self, Hmlpt):
    for n in range(self.nParticles):
        tau = self.cholesky_identity_add_individual(Hmlpt[:, :, n])
        Hmlpt[:, :, n] = Hmlpt[:, :, n] + tau * np.identity(self.DoF)
    return Hmlpt


def plot_linesearch_ensemble(self, amax, iter):  # , array):
    step_size = np.arange(0, amax, .01)
    phi = np.vectorize(self.phi_for_ensemble)
    cost = phi(step_size)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('stepsize a')
    ax.set_ylabel('Cost')
    ax.set_title('Armijo line search', fontdict={'fontsize': 20})
    ax.plot(step_size, cost)
    fig.show()
    # for val in array:
    #     textstr += '%f,\n' % (val)
    #
    # # matplotlib.patch.Patch properties
    # props = dict(boxstyle='square', facecolor='green', alpha=0.5)
    #
    # # place a text box in upper left in axes coords
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    #
    fig.savefig('armijo_linesearch_iter_%i.png' % iter)

    return fig


def linesearch_scheme(self, amax, maxiter):
    info('Searching for stepsize...')
    try:
        candidate_step = \
            scalar_search_wolfe2(self.phi_for_ensemble, nd.Derivative(self.phi_for_ensemble), maxiter=maxiter,
                                 amax=amax)[0]
    except OverflowError:
        log.info(OverflowError)
        pass
    if candidate_step is not None:
        self.stepsize = candidate_step
        log.info('Stepsize found: %f' % self.stepsize)
    else:
        if iter_ != 0:
            log.info('No stepsize found, dividing previous stepsize by 2...')
            # self.stepsize /= 2.
            self.stepsize = 0.5
            candidate_step = self.stepsize
        else:
            log.info('No stepsize found, setting stepsize = 1 for this case')
            self.stepsize = 1
            candidate_step = self.stepsize

            log.info('Time taken to find stepsize: %.4f, maxiter = %i, step = %f' % (
                time.time() - start, maxiter, candidate_step))


def plot_derivative_linesearch(self, amax, array):
    n = self.step_n
    step_size = np.arange(0, amax, .01)
    dphi = np.vectorize(nd.Derivative(self.phi_for_particle))
    dcost = dphi(step_size)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('stepsize a')
    ax.set_ylabel('Cost')
    ax.set_title('Armijo derivative line search: Particle %i' % self.step_n, fontdict={'fontsize': 20})
    ax.plot(step_size, dcost)
    textstr = ''
    for val in array:
        textstr += '%f,\n' % (val)

    # matplotlib.patch.Patch properties
    props = dict(boxstyle='square', facecolor='green', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    fig.savefig('armijo_linesearch_derivative-%i.png' % n)

    ###############################
        # for n in range(self.nParticles):
        #     if n == 0:
        #         Q_loop = np.zeros((self.DoF, 1))
        #         # Q = np.tensordot(self.alphas[:, n], self.k_gram[:, n], axes=0).reshape((self.DoF, 1))
        #         for l in range(self.nParticles):
        #             Q_loop += (self.alphas[:, l] * self.k_gram[l, n]).reshape((self.DoF, 1))
        #     else:
        #         Q_loop = np.zeros((self.DoF, 1))
        #         # Q = np.hstack((Q, np.tensordot(self.alphas[:, l], self.k_gram[l, n], axes=0).reshape(self.DoF, 1)))
        #         for l in range(self.nParticles):
        #             Q_loop += (self.alphas[:, l] * self.k_gram[l, n]).reshape((self.DoF, 1))


    def buildGradUpdate(self):
        d = np.zeros((self.DoF, self.DoF))
        for l in range(self.nParticles):
            d += np.outer(self.alphas[:, l], self.gradient_k_gram[l, 0])

        pass

        # self.gradient_k_gram[:, m, :].T.dot(self.gradient_k_gram[:, n, :]))
    def createLineSearchCrossSection(self):
        phi = lambda eps: np.mean(self.model.getMinusLogPosterior_ensemble(self.particles + eps * self.Q) - np.log(np.det(np.identity(self.DoF) + eps * self.gradUpdate)))


buildmgJ

# post = self.getGradientMinusLogPosterior_ensemble(thetas)
# Debugging
# mgJ_loop = np.zeros((DoF, 1))
# for l in range(nParticles):
#     mgJ_loop += (k_gram[0, l] * post[:, l] + grad_k_gram[0, l, :]).reshape((DoF, 1))
#
# mgJ_loop = mgJ_loop / nParticles
#
# x = np.einsum('nm, dn -> dm', k_gram[:, :], self.getGradientMinusLogPosterior_ensemble(thetas))
# y = np.mean(grad_k_gram, axis=1).T
# result = x / nParticles + y

# self.mgJ_ensemble = \
#     (-1 * np.einsum('nm, dn -> dm', k_gram[:, :],self.getGradientMinusLogPosterior_ensemble(thetas)) / nParticles
#                      + np.mean(grad_k_gram, axis=1).T).flatten(order='F').reshape((nParticles * DoF, 1))
#
#
#
# New


# Old
# if self.optimizeMethod == 'SVN':
#     self.mgJ_ensemble = (-1 * np.einsum('nm, dn -> dm', k_gram[:, :], self.getGradientMinusLogPosterior_ensemble(thetas)) / nParticles\
#                         + np.mean(grad_k_gram, axis=1).T).flatten(order = 'F').reshape((nParticles * DoF, 1))
# elif self.optimizeMethod == 'SVGD':
#     self.mgJ_ensemble = (-1 * np.einsum('nm, dn -> dm', k_gram[:, :], self.getGradientMinusLogPosterior_ensemble(thetas)) / nParticles\
#                          + np.mean(grad_k_gram, axis=1).T)


pass
# TODO
# (1) hessian evaluation matrix?
# (2) evaluate h_ij_nm
# (3) insert the object for a specific n


def newDrawFromPrior(self):
    pass

# not needed
# self.stepsize = 1 # Default
# self.particleRange = particleRange
# self.startPositions = startPositions

# if startPositions is not None:
#     self.particles = copy.deepcopy(startPositions)  # Manually insert starting positions
#     self.nParticles = self.particles.size
# elif startPositions is None:
#     self.resetParticles(self.particleRange)
#     self.startPositions = copy.deepcopy(self.particles)
# self.stochastic_perturbation = 0

# Inherited variables needed
# Linesearch
amax = 1
maxiter = 100
# self.step_n = i_
# self.stepsize = 0.5 # GOOD FOR SVVG
# self.stepsize = 0.1 # GOOD FOR SVN

def evaluate_mgJ(self, particle, distribution, gmlpt):
    """
    Calculates kernel and evaluates the minus gradient of cost function at particle i
    Args:
        **kwargs:
        particle : particle at which to evaluate (int)
        distribution: distribution around and inlcuding particle i (d x n array)
        gmlpt: minus gradient of cost functional minus log posterior given distribution (d x n array)

    Returns: Minus gradient of cost function at particle i, given the distribution around i.

    """
    dxs = distribution[:, particle, np.newaxis] - distribution

    # distances = np.apply_along_axis(np.linalg.norm, 0, dxs)

    M_dxs = np.matmul(self.metric, dxs)

    dxT_M_dxs = np.sum(dxs * M_dxs, 0)

    # h = np.median(np.sqrt(dxT_M_dxs)) ** 2 / np.log(self.nParticles) # Revised Kernel bandwith
    # h1 = np.median(np.sqrt(dxT_M_dxs)) / np.log(self.nParticles) # Revised Kernel bandwith
    h = 1
    d = (2 * self.DoF)  # Old Kernel bandwith
    h2 = (h + d) / 2
    kerns = np.exp(- 1 / h * dxT_M_dxs)

    gkerns = 2 * M_dxs * kerns * (1 / h)  # added in 1/h term in derivative

    mgJ = np.mean(- gmlpt * kerns + 2 * gkerns, 1)

    return mgJ, kerns, gkerns

# working linesearch
    def phi(self, eps):
        I = np.eye(self.DoF)
        sum = 0

        cost_repulsion_sum = 0
        cost_shape_sum = 0

        for n in range(self.nParticles):
            x = self.particles[:, n]
            Q = self.Q[:, n]
            gradQ = self.grad_w(n)
            cost_shape_term = self.getMinusLogPosterior_individual(x + eps * Q)
            cost_repulsion_term = - np.log(np.abs(np.linalg.det(I + eps * gradQ)))

            cost_shape_sum += cost_shape_term
            cost_repulsion_sum += cost_repulsion_term
            pass
            # sum += cost_shape + cost_repulsion


        # log.info('cost shape = %f' % (cost_shape_sum / self.nParticles))
        # log.info('cost_repulsion = %f' % (cost_repulsion_sum / self.nParticles))
            # sum += self.getMinusLogPosterior_individual(x + eps * Q) - np.log(np.abs(np.linalg.det(I + eps * gradQ)))
        # return sum / self.nParticles

        cost_shape_total = cost_shape_sum / self.nParticles
        cost_repulsion_total = cost_repulsion_sum / self.nParticles

        return cost_shape_total, cost_repulsion_total


    @profile
    def buildBlockMatrix(self):

        for m in range(self.nParticles): # Rows
            for n in range(self.nParticles): # Cols
                if n == 0:
                    # first_block = calculateBlockLoop(GN_Hmlpt, self.DoF, self.nParticles, m, n, self.k_gram, self.gradient_k_gram)
                    first_block = self.h_ij(m, n)
                    row = first_block
                else:
                    # next_block = calculateBlockLoop(GN_Hmlpt, self.DoF, self.nParticles, m, n, self.k_gram, self.gradient_k_gram)
                    next_block = self.h_ij(m, n)
                    row = np.hstack((row, next_block))
            if m == 0:
                H = row
            else:
                H = np.vstack((H, row))
        # End
        # self.H = H
        # Testing a hypothesis:
        self.H = H


    ### Textbox settings ###############################################################
    # textstr = ''
    # for tuple in attributes_list:
    #     if tuple[0] == 'nIterations':
    #         nIterations = tuple[1]  # Pick up this value for number of frames in animation
    #     if tuple[0] != 'filename':
    #         if tuple == attributes_list[-1]:
    #             textstr += '%s: %s' % (tuple[0], tuple[1])
    #         else:
    #             textstr += '%s: %s \n' % (tuple[0], tuple[1])
    #
    # # matplotlib.patch.Patch properties
    # props = dict(boxstyle='square', facecolor='green', alpha=0.5)
    #
    # # place a text box in upper left in axes coords
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    # for tuple in attributes_list:
    #     if tuple[0] == 'nIterations':
    #         nIterations = tuple[1] # Pick up this value for number of frames in animation
    #     if tuple[0] != 'filename':
    #         if tuple == attributes_list[-1]:
    #             textstr += '%s: %s' % (tuple[0], tuple[1])
    #         else:
    #             textstr += '%s: %s \n' % (tuple[0], tuple[1])
    #
    # # matplotlib.patch.Patch properties
    # props = dict(boxstyle='square', facecolor='green', alpha=0.5)
    #
    # # place a text box in upper left in axes coords
    # # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    # #         verticalalignment='top', bbox=props)
    # ####################################################################################
    # Access metadata and store in list of tuples from particle_history_path
    # with h5py.File(particle_history_path, 'r') as hf:
    #     group = hf["particle_history"]
    #     attributes_list = list(group.attrs.items())
    #     # Used to feed into FuncAnimation
    #     history = []
    #     group.visit(history.append)
    #     iterationsPerformed = len(history)
    #     particles_x_0 = hf["particle_history/%s" % history[0]][0, :]
    #     particles_y_0 = hf["particle_history/%s" % history[0]][1, :]
    # tparticles_x_0 = composite_map[0]['T'][:, 0]
    # tparticles_y_0 = composite_map[0]['T'][:, 1]

        # with h5py.File(particle_history_path, 'r') as hf:
        #     group = hf["particle_history"]
        #     history = []
        #     group.visit(history.append)
        #     particles_x_i = hf["particle_history/%s" % history[i]][0, :]
        #     particles_y_i = hf["particle_history/%s" % history[i]][1, :]
        #     scat.set_offsets(np.c_[particles_x_i, particles_y_i])
    # @profile
    # def buildmgJ(self):
    #     """
    #     Stores n*d x 1 array of derivatives of mgJ evaluated at ensemble
    #     Returns: nothing
    #
    #     """
    #     result = -1 * np.einsum('mn, om -> on', self.k_gram, self.gmlpt) / self.nParticles + np.mean(self.gradient_k_gram, axis=0).T
    #
    #     if self.optimizeMethod == 'SVN':
    #         result = result.flatten(order = 'F').reshape((self.nParticles * self.DoF, 1))
    #
    #     self.mgJ_ensemble = result
    #     #

#%%
sum = 0
for i in range(3):
    sum += i
print(sum)
# def buildHessianKernel(self):
#     self.hessKernel = 2 * self.M / self.bandwidth * (np.identity(self.DoF) * self.k_gram - self.delta_xs * self.gradient_k_gram)
#     pass
# def grad_w_test(self, m):
#     """
#     This measures how the flow direction for particle m changes as we independently vary the position of particle m.
#     Args:
#         m: particle number
#
#     Returns: d x d matrix
#
#     """
#     return -1 * np.einsum('dn, nmb -> dbm', self.alphas, self.gradient_k_gram_test)[:, :, m]
# @profile


# def backtrackLinesearch(self):
#     """
#     Chooses a stepsize
#     Returns: stepsize satisfying given criteria
#
#     """
#     step = 1
#     maxiter = 4
#     counter=0
#
#     # will be used later on in phi(eps)
#     self.gradQ = np.zeros((self.nParticles, self.DoF, self.DoF))
#     for n in range(self.nParticles):
#         self.gradQ[n] = self.grad_w(n)
#         # we wont need this, because all linesearch will be done on the test set!!!
#         #self.gradQ_test
#     self.phi_before = self.phi(0)
#     log.info('phi_before: %f' % self.phi_before)
#     # phi_step = self.phi(step)
#     while self.phi(step) > self.phi_before: # or (self.phi(step)[0] + self.phi(step)[1]) > phi_shape:
#         counter += 1
#         step /= 2
#         if counter > maxiter:
#             self.converged = True
#             return step
#
#
#     #self.phi_before = self.phi(step)
#     return step

# @profile
# def phi(self, eps): # phi_test is the only one that needs to be used!!!
#     I = np.eye(self.DoF)
#     cost_shape = 0
#     determinants = np.zeros(self.nParticles)
#     for n in range(self.nParticles):
#         x = self.particles[:, n]
#         Q = self.Q[:, n]
#         cost_shape += self.getMinusLogPosterior_individual(x + eps * Q)
#         determinants[n] = (np.linalg.det(I + eps * self.gradQ[n]))
#         # cost_repulsion += - np.log(np.abs(np.linalg.det(I + eps * gradQ)))
#     if np.any(np.sign(determinants) != np.sign(determinants[0])):
#         return np.inf
#
#     cost_repulsion = - np.sum(np.log(np.abs(determinants)))
#     return (cost_shape + cost_repulsion) / self.nParticles

# @profile
# def grad_w(self, m):
#     """
#     This measures how the flow direction for particle m changes as we independently vary the position of particle m.
#     Args:
#         m: particle number
#
#     Returns: d x d matrix
#
#     """
#     return -1 * np.einsum('dn, nmb -> dbm', self.alphas, self.gradient_k_gram)[:, :, m]


# @profile
# def solveSystem(self):
#     """
#     Solves newton system with CG
#     Returns: d x n matrix for alphas
#
#     """
#     self.alphas = scipy.sparse.linalg.cg(self.H, self.mgJ_ensemble )[0].reshape((self.DoF, self.nParticles), order = 'F')

# def getMoments(self, thetas):
#     """
#     Calculates first two moments of the ensemble
#     Args:
#         thetas: ensemble n x d
#
#     Returns: two d-dimensional arrays
#
#     """
#     mean = np.mean(thetas, axis=0)
#     variance = np.var(thetas, axis=0)
#     return mean, variance


# @profile

# # @profile
# def h_ij(self, m, n):
#     """
#     Calculates block for given m, n pair. ~7x faster than naive sum!!!
#     Args:
#         m:
#         n:
#
#     Returns:
#
#     """
#
#     return (np.einsum('dem, m, m -> de', self.GN_Hmlpt, self.k_gram[:, m], self.k_gram[:, n]) + self.gradient_k_gram[:, m, :].T.dot(
#         self.gradient_k_gram[:, n, :])) / self.nParticles

# def buildBlockMatrix(self):
#     """
#     Builds full Hessian for SVN
#     Returns: nd x nd array
#
#     """
#     dim = self.nParticles * self.DoF
#     H = np.zeros((dim, dim))
#     for m, n in itertools.product(range(self.nParticles), range(self.nParticles)):
#         H[m * self.DoF : self.DoF * (m + 1), n * self.DoF : self.DoF * (n + 1)] = self.h_ij(m, n)
#     self.H = H
#

# def buildGradientGramKernel_test(self):
#     """
#     Stores n x n x d matrix of kernel derivatives.
#     Args:
#         distribution: ensemble positions d x n
#
#     Returns: nothing
#
#     """
#     self.gradient_k_gram = -2 * np.einsum('li,ijk', self.M, self.delta_xs_test) * self.k_gram_test[
#             ..., np.newaxis] / self.bandwidth

# def buildGradientGramKernel_new(self):
#     gradient_k_gram = -2 * contract('li,ijk',self.M, self.delta_xs) * self.k_gram[..., np.newaxis] / self.bandwidth
#     return gradient_k_gram

# @profile
# def buildGradientGramKernel(self):
#     """
#     Stores n x n x d matrix of kernel derivatives.
#     Args:
#         distribution: ensemble positions d x n
#
#     Returns: nothing
#
#     """
#     self.gradient_k_gram = -2 * np.einsum('li,ijk',self.M, self.delta_xs) * self.k_gram[..., np.newaxis] / self.bandwidth


# def getKernel(self, ensemble1, ensemble2):
#     if ensemble1.shape[0] == self.DoF:
#         ensemble1 = ensemble1.T
#         ensemble2 = ensemble2.T


# @profile
# def buildGramKernel(self):
#     """
#     Stores the gram matrix (n x n), the inter-particle distances (d x n x n), and the support scaling (scalar)
#     Returns: Nothing
#
#     """
#
#     # self.metric_ensemble = self.metricActionEnsemble_new()
#     self.metric_ensemble = self.metric_action_ensemble()
#     # self.metric_ensemble_test = self.metric_action_ensemble()
#     median = np.median(np.trim_zeros(np.sqrt(self.metric_ensemble).flatten()))
#     # self.bandwidth = 1
#     self.bandwidth = median ** 2 / np.log(self.nParticles)
#     self.k_gram = np.exp(-1 * self.metric_ensemble / self.bandwidth)

# def metricActionEnsemble_opt(self):
#     """
#     Much faster way of computing metric action over ensemble
#     Returns: n x n metric action on ensemble. sets deltas as well (n x n x d array)
#
#     """
#     # Beginning with thetas -> n x d array
#     # metric_action_ensemble = np.zeros(self.nParticles, self.nParticles)
#
#     a = self.trans_particles
#
#     delta = (a - a[:, np.newaxis]).reshape(-1, a.shape[1]).reshape((self.nParticles, self.nParticles, self.DoF))
#     return contract('mnb, bd, mnd -> nm', delta, self.M, delta)

# def getMetricDeltas(self, M, deltas):

# @profile
# def metric_action_ensemble(self):
#     """
#     calculate M(x, y) = (x-y).T @ metric @ (x-y) for all (x, y) in ensemble
#     Returns: n x n np.array
#
#     """
#     for particle in range(self.nParticles):
#         dxs = self.particles[:, particle, np.newaxis] - self.particles
#         # dxs_test_build = xxx
#         # particle_distances = np.concatenate(particle_distances, particle_distance_n, axis=2)
#         self.M_dxs = np.matmul(self.M, dxs)
#         dxsT_M_dxs = np.sum(dxs * self.M_dxs, 0)  # Norm squared
#         if particle == 0:
#             delta_xs = dxs.reshape((self.DoF, self.nParticles, 1))
#             metric_action_ensemble = dxsT_M_dxs
#         else:
#             delta_xs = np.concatenate((delta_xs, dxs.reshape((self.DoF, self.nParticles, 1))), axis=2)
#             metric_action_ensemble = np.vstack((metric_action_ensemble, dxsT_M_dxs))
#     self.delta_xs = np.swapaxes(delta_xs, 1, 2) # (d x n x n matrix). Ie, delta_xs[:, 2, 5] gives the radial vector separation r2-r5
#     return metric_action_ensemble

# def T(self, bandwidth, metric, particles, alphas, Q = None, stepsize = None):
#     pass
# def T_composition(self, dictionary):


# @profile
# def getMinusLogPosterior_ensemble(self, thetas):
#     """
#     Evaluates Posterior at each particle position
#     Args:
#         thetas: ensemble locations (dxN matrix)
#
#     Returns: 1xN matrix
#
#     """
#
#     return np.apply_along_axis(self.model.getMinusLogLikelihood_individual, 0, thetas) + \
#            np.apply_along_axis(self.model.getMinusLogPrior_individual, 0, thetas)
#
# # @profile
# def getGradientMinusLogPosterior_ensemble(self, thetas):
#     """
#     Method to get gradient of negative log posterior
#     Args:
#         thetas: array of all particle positions (d x n array)
#
#     Returns: array of gradients evaluated at each particle position (d x n array)
#
#     """
#     # return self.getGradientMinusLogPrior(thetas) + self.getGradientMinusLogLikelihood(thetas)
#     return np.apply_along_axis(self.model.getGradientMinusLogLikelihood_individual, 0, thetas) + \
#            np.apply_along_axis(self.model.getGradientMinusLogPrior_individual, 0, thetas)
#
# # @profile
# def getGNHessianMinusLogPosterior_ensemble(self, thetas):
#     """
#     Method to get hessian of negative log posterior
#     Args:
#         thetas: array of all particle positions (d x n array)
#
#     Returns: array of hessians evaluated at each particle position (d x d x n array)
#
#     """
#     return (np.apply_along_axis(self.model.getGNHessianMinusLogPrior_individual, 0, thetas) + \
#            np.apply_along_axis(self.model.getGNHessianMinusLogLikelihood_individual, 0, thetas)).reshape((self.DoF,self.DoF, self.nParticles))


# def setMomentChanges(self, iter_, X):
#     """
#     Takes care of ensemble moment calculations, and stores percent change criteria
#     Args:
#         iter_: current iteration
#
#     Returns: Nothing. Sets mean, variance, and percent changes.
#
#     """
#     if iter_ == 0:
#         self.mean_percent_change, self.variance_percent_change = np.infty, np.infty
#         self.mean, self.variance = self.getMoments(X)
#         self.mean_0_norm, self.variance_0_norm = np.linalg.norm(self.mean), np.linalg.norm(self.variance)
#     if iter_ > 0:
#         old_mean = self.mean
#         old_variance = self.variance
#         self.mean, self.variance = self.getMoments(X)
#         self.mean_percent_change, self.variance_percent_change = self.percentChange(old_mean, old_variance)


# def percentChange(self, old_mean, old_variance):
#     """
#     Calculates percent change in moments
#     Args:
#         old_mean:
#         old_variance:
#
#     Returns: The percent change.
#
#     """
#     a = 100 * np.linalg.norm(old_mean - self.mean) / self.mean_0_norm
#     b = 100 * np.linalg.norm(old_variance - self.variance) / self.variance_0_norm
#     return a, b

# def checkEnsembleMomentsConverge(self):
#     """
#     Checks if the moments fall below a tolerance
#     Returns: Boolean
#
#     """
#     tol = .1
#     if self.mean_percent_change < tol and self.variance_percent_change < tol:
#         return True
#     else:
#         return False

# composite_transport_map[iter_]['X'] = np.copy(X)
# test_set_history[iter_]['list'] = copy.deepcopy(test_sets)
# composite_transport_map[iter_]['T'] = np.copy(T)

# storeIterationDataH5(iter_, self.nIterations, self.history_path, self.particles, self.mean, self.variance)

# for m in range(self.nTestSets):
#     # make sure whole individual and piecewise match
#     km = self.getKernelWhole(bandwidth=h, metric=metric_new, X=X, T_list=[test_sets[m]])
#     deltas_Tm = self.getDeltas(X, test_sets[m])
#     metricDeltas_Tm = self.getMetricDeltas(metric_new, deltas_Tm)
#     deltaMetricDeltas_Tm = self.getDeltasMetricDeltas(deltas_Tm, metricDeltas_Tm)
#     km_pw = self.getKernelPiecewise(bandwidth=h, deltasMetricDeltas=deltaMetricDeltas_Tm)
#     assert(np.allclose(km, km_pw))

# X += eps * wx


# test_set_history[iter_ + 1]['%i' % n] = test_set_history[iter]['%i' % n] + eps * self.w(alphas, kots[t])

# for t in range(self.nTestSets):
#     if
#     test_set_history[iter_ + 1]['%i' % t] = test_set_history[iter_]['%i' % t] + eps * self.w(alphas, kots[t])
# T += eps * wt


# for m in range(self.nTestSets):
#     # make sure piecewise and list kernel method calculated earlier work
#     if m != test_set_choice:
#         deltas_Tm = self.getDeltas(X, test_sets[m])
#         metricDeltas_Tm = self.getMetricDeltas(metric_new, deltas_Tm)
#         deltaMetricDeltas_Tm = self.getDeltasMetricDeltas(deltas_Tm, metricDeltas_Tm)
#         km_pw = self.getKernelPiecewise(bandwidth=h, deltasMetricDeltas=deltaMetricDeltas_Tm)
#         assert(np.allclose(kots[m], km_pw))
#     elif m==test_set_choice:
#         assert(kots[m] == None)
# assert(np.allclose(kots[1], k1_pw))
# assert(np.allclose(kots[2], k2_pw))


# for t, ots in enumerate(test_sets):
#     if t == test_set_choice:
#         test_sets[t] += eps * wt
#     elif t != test_set_choice:
#         test_sets[t] += eps * self.w(alphas, kots[t])


# # Begin Calculations For Stein
# if self.optimizeMethod == 'SVN':
#     self.GN_Hmlpt = self.getGNHessianMinusLogPosterior_ensemble(self.particles)
#     self.M = np.mean(self.GN_Hmlpt, axis=2)
# elif self.optimizeMethod == 'SVGD':
#     self.M = np.eye(2)
# # M = np.eye(2)
# # assert (self.check_symmetric(self.M))
#
# log.info('Finding search direction')
#
# # Always need to calculate these
# self.buildGramKernel() # make a function of first and second ensemble
# self.buildGradientGramKernel() # make a function of first and second ensemble
#
# # only need gradients once per iteration
# # if
# self.gmlpt = self.getGradientMinusLogPosterior_ensemble(self.particles)
# self.buildmgJ()
#
# if self.optimizeMethod == 'SVN':
#     self.buildBlockMatrix()
#     np.random.seed(1)
#     self.solveSystem()
#
# self.buildEnsembleUpdate()
#
# if isSVN:
#     self.stepsize = self.backtrackLinesearch()
#     # if iter_ <= 15:
#     #     # self.stepsize = 0.1
#     #     self.stepsize = 0.2
#     # elif iter_ > 15:
#     #     self.stepsize = 0.2
#     # self.stepsize = 0.1
# elif isSVGD:
#     self.stepsize = 0.01
#     # if iter_ <= 20:
#     #     self.stepsize = 0.5
#     # elif iter_ > 20:
#     #     self.stepsize = 0.25
#
# if self.converged == True:
#     pass
#     log.info('converged = %s' % self.converged)
#     break
#
# # self.stepsize = 1
# # self.stepsize = 0.005 # GOOD FOR SVN
#
# log.info('Stepsize used: %f' % self.stepsize)
# log.info('Step norm: %f' % np.mean(np.linalg.norm(self.alphas,axis=0)))
# self.particles += self.stepsize * self.Q
# # self.particles = T(self.stepsize, bandwidth, metric, self.particles, alphas, Q = None)
# # self.particles_test += self.stepsize * self.Q_test
# # compute statistics right here!!!


def metricActionEnsemble_new(self):
    """
    Much faster way of computing metric action over ensemble
    Returns: n x n metric action on ensemble. sets deltas as well (n x n x d array)

    """
    # Beginning with thetas -> n x d array
    # metric_action_ensemble = np.zeros(self.nParticles, self.nParticles)
    a = self.particles
    if a.shape[0] == self.DoF:
        a = self.particles.T
    delta = (a - a[:, np.newaxis]).reshape(-1, a.shape[1]).reshape((self.nParticles, self.nParticles, self.DoF))
    return np.einsum('mnb, bd, mnd -> nm', delta, self.M, delta)

    def getPreconditioner(self, A, M, k=0):
        """Extracts blocks of size M from the kth diagonal
        of square matrix A, whose size must be a multiple of M."""

        # Check that the matrix can be block divided
        if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
            raise StandardError('Matrix must be square and a multiple of block size')

        # Assign indices for offset from main diagonal
        if abs(k) > M - 1:
            raise StandardError('kth diagonal does not exist in matrix')
        elif k > 0:
            ro = 0
            co = abs(k) * M
        elif k < 0:
            ro = abs(k) * M
            co = 0
        else:
            ro = 0
            co = 0

        invblocks = np.array([np.linalg.inv(A[i + ro:i + ro + M, i + co:i + co + M])
                           for i in range(0, len(A) - abs(k) * M, M)])

        return scipy.sparse.block_diag(invblocks)

#%%


# for p in range(1, 4):
#
#     ## Read in picture
#     fname = "heatflow%03d.png" %p
#     img = mgimg.imread(fname)
#     imgplot = plt.imshow(img)
#
#     # append AxesImage object to the list
ims = []
fig = plt.figure()
w_in_inches = 4
h_in_inches = 4
fig.set_size_inches(w_in_inches, h_in_inches, True)
for i in range(iter):
    name = 'agsehrdrhtej_%i.jpeg' % i
    data_i = pd.DataFrame(composite_map[i]['X'])
    data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
    g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
    # plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
    g.fig.savefig(name)
#%%
fig, ax = plt.subplots()
fig.set_size_inches(w_in_inches, h_in_inches, True)
dpi = 180
def update(i=0):
    name = 'agsehrdrhtej_%i.jpeg' % i
    img = Image.open(name)
    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
ani = matplotlib.animation.FuncAnimation(fig, update, frames=iter, repeat=True)
ani.save('a.gif', writer='imagemagick', dpi=dpi)

#%%






#%%
# img = mgimg.imread(name)
# imgplot = plt.imshow(img)
# ims.append([imgplot])
# ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
# ani.save('a.gif', writer='imagemagick')
#%%
# for i in range(iter):
#     name = 'agsehrdrhtej_%i.png' % i
#     send2trash(name)

# def update(i=0):
#     # .. update your data ..
#     data_i = pd.DataFrame(composite_map[i]['X'])
#     data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
#     if 'g' in locals():
#         plt.close(g.fig)
#     g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
#     plt.pause(0.01)
#
# anim = matplotlib.animation.FuncAnimation(fig, update, frames=20, repeat=False)
# anim.save('a.gif', writer='imagemagick')

class animate_kde:
    def __init__(self):
        self.anim = None
        CWD = os.getcwd()
        rel_file_path='/drivers/outdir/1596598591/rosenbrock_proper_nP_5000_1000_1596598591.h5'
        abs_file_path = CWD + rel_file_path
        self.composite_map = dd.io.load(abs_file_path)
        self.final_iter =
        self.changeFigInCamera(0)
        self.camera = Camera(self.fig)

    def changeFigInCamera(self, i):
        data_i = pd.DataFrame(self.composite_map[i]['X'])
        data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
        g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
        plt.subplots_adjust(top=0.9)
        plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
        g.ax_marg_x.set_title('stein Ensemble KDE Iteration: %i' %i)
        self.fig = g.fig

    def animate(self):
        # for i in range(self.final_iter):
        for i in range(3):
            if 'jp' in locals():
                plt.close(jp.fig)
            self.changeFigInCamera(i)
            self.camera.snap()
        anim = self.camera.animate(blit=False)
        anim.save('a.gif', writer='imagemagick')



# akde = animate_kde()
# akde.animate()

# #%%
# sns.set_style('ticks')
# g = sns.jointplot(x='x', y='y', data=data, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)

# g.fig.show()
#%%
# sns.set_style('white')
# x = composite_map[final_iter]['X'][:, 0]
# y = composite_map[final_iter]['X'][:, 1]
# g = sns.JointGrid(x=x, y=y, height=4, space=0, xlim=(-2, 2), ylim=(-2, 2))
# plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
# g.plot_joint(sns.kdeplot, shade=True, joint_kws={"cut": 15})
# g.plot_marginals(sns.kdeplot, shade=True)
# g.fig.show()

#
# def update(i):
#     data_i = pd.DataFrame(composite_map[i]['X'])
#     sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
#     plt.subplots_adjust(top=0.9)
#     plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
#     # g.ax_marg_x.set_title('stein Ensemble KDE Iteration: %i' %i)
#
# # Setup static figure
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.set_xlabel('mass_1')

# class Heatmap:
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
#         self.anim = None
#
#     def animate(self):
#         def init():
#             sns.heatmap(np.zeros((10, 10)), vmax=.8, ax=self.ax)
#
#         def animate(i):
#             self.ax.texts = []
#             sns.heatmap(np.random.rand(10, 10), annot=True, vmax=.8, cbar=False, ax=self.ax)
#
#         self.anim = matplotlib.animation.FuncAnimation(self.fig, animate, init_func=init, frames=20, repeat=False)
#
# hm = Heatmap()
# hm.animate()
# hm.anim.save('a.gif', writer='imagemagick')

# ax.set_ylabel('mass_2')
# ax.set_title('SVN particle flow', fontdict={'fontsize': 20})
#
# frames = sorted(list(composite_map.keys()))[-1]
# anim = FuncAnimation(fig, update, interval=3000 / iterationsPerformed, frames=iterationsPerformed,
#                      repeat_delay=2000)
#
# anim.save('kde_animation.gif')

# def init(i=0):
#     data_i = pd.DataFrame(composite_map[0]['X'])
#     data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
#     g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
#     plt.subplots_adjust(top=0.9)
#     plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
#     g.ax_marg_x.set_title('stein Ensemble KDE Iteration: %i' %i)
#
#
# def update(i):
#     data_i = pd.DataFrame(composite_map[i]['X'])
#     data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
#     g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
#     plt.subplots_adjust(top=0.9)
#     plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
#     g.ax_marg_x.set_title('stein Ensemble KDE Iteration: %i' %i)




# fig = plt.figure()
# camera = Camera(fig)
# for i in range(3):
#     if 'jp' in locals():
#         plt.close(jp.fig)
#     data_i = pd.DataFrame(composite_map[i]['X'])
#     data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
#     g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(-2, 2), ylim=(-2, 2), space=0)
#     plt.subplots_adjust(top=0.9)
#     plt.title('Include this transparent title because otherwise marginals and joint wont have space=0.', y=1.3, fontsize = 16, alpha=1e15)
#     g.ax_marg_x.set_title('stein Ensemble KDE Iteration: %i' %i)
#     camera.snap()
#     plt.pause(0.01)
# anim = camera.animate(blit=False)
# anim.save('a.gif', writer='imagemagick')




# def get_data(i=0):
#     g.x, g.y = get_data(i)
#     g.plot_joint(sns.kdeplot, cmap="Purples_d")
#     g.plot_marginals(sns.kdeplot, color="m", shade=True)
#     i = int(i)
#     x = composite_map[i]['X'][:, 0]
#     y = composite_map[i]['X'][:, 1]
#     return x,y
#
#
# x,y = get_data()
# g = sns.JointGrid(x=x, y=y, height=4)
# lim = (-2, 2)
# def prep_axes(g, xlim, ylim):
#     g.ax_joint.clear()
#     g.ax_joint.set_xlim(xlim)
#     g.ax_joint.set_ylim(ylim)
#     g.ax_marg_x.clear()
#     g.ax_marg_x.set_xlim(xlim)
#     g.ax_marg_y.clear()
#     g.ax_marg_y.set_ylim(ylim)
#     plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)
#     plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)
#     plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=False)
#     plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=False)
#     plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=False)
#     plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=False)
#     plt.setp(g.ax_marg_x.get_yticklabels(), visible=False)
#     plt.setp(g.ax_marg_y.get_xticklabels(), visible=False)
#
# def animate(i):
#     g.x, g.y = get_data(i)
#     prep_axes(g, lim, lim)
#     g.plot_joint(sns.kdeplot, cmap="Purples_d")
#     g.plot_marginals(sns.kdeplot, color="m", shade=True)
#
# # frames=np.sin(np.linspace(0,2*np.pi,17))*5
# ani = matplotlib.animation.FuncAnimation(g.fig, animate, frames=frames, repeat=True)
# ani.save('a.gif', writer='imagemagick')


# def getBandwidthBM(self, gradLogDensity, X):
#     get_obj_ibandw = lambda ibw: self.getMMD(X + self.getGradLogDensity(), Y)
#
#     explore_ratio = 1.1
#     obj_ibw_in = get_obj_ibandw(ibw_in)
#     epsi = 1e-6
#     grad_ibw_in = (get_obj_ibandw(ibw_in+epsi)-obj_ibw_in)/epsi
#     if grad_ibw_in<0:
#         ibw_1 = ibw_in*explore_ratio
#     else:
#         ibw_1 = ibw_in/explore_ratio
#
#     obj_ibw_1 = get_obj_ibandw(ibw_1)
#     slope_ibw = (obj_ibw_1-obj_ibw_in)/(ibw_1-ibw_in)
#     ibw_2 = (ibw_in * slope_ibw - 0.5 * grad_ibw_in * (ibw_1 + ibw_in)) / (slope_ibw - grad_ibw_in)
#     obj_ibw_2 = get_obj_ibandw(ibw_2)
#     if ~isnan(ibw_2)&&ibw_2>0
#         if obj_ibw_1<obj_ibw_in
#             if obj_ibw_2<obj_ibw_1
#                 ibw_out = ibw_2;
#             else
#                 ibw_out = ibw_1;
#
#         else
#             if obj_ibw_2<obj_ibw_in
#                 ibw_out = ibw_2;
#             else
#                 ibw_out = ibw_in;
#
#     else
#         if obj_ibw_1<obj_ibw_in
#             ibw_out = ibw_1;
#         else
#             ibw_out = ibw_in;


    def naive_longrange_bandwidth(self, X, T, h0, gmlpt_X, metric_new, metricDeltas, deltaMetricDeltas):
        tau = 0.01
        hi0 = 1 / h0

        gmlpt_T = self.getGradientMinusLogPosterior_ensemble_new(T)
        deltas_T = self.getDeltas(X, T)
        metricDeltas_T = self.getMetricDeltas(metric_new, deltas_T)
        deltaMetricDeltas_T = self.getDeltasMetricDeltas(deltas_T, metricDeltas_T)

        def cost(hi):
            h = 1 / hi
            try:
                assert(h > 0)
                kx = self.getKernelPiecewise(h, deltaMetricDeltas)
                gkx = self.getGradKernelPiecewise(h, kx, metricDeltas)
                kt = self.getKernelPiecewise(h, deltaMetricDeltas_T)
                gkt = self.getGradKernelPiecewise(h, kt, metricDeltas_T) # (Does this need a minus sign???) No, its in the method!
                X_p = X + tau * self.mgJ_new(kx, gkx, gmlpt_X)
                T_p = T + tau * self.mgJ_new(kt, gkt, gmlpt_T)
                return self.getMMD(X_p, T_p, mode='BM')
            except:
                log.info('Evaluating cost failed. Returning infty')
                return np.infty
        explore_ratio = 1.01 # used to be 1.1
        cost0 = cost(hi0)
        eps = 1e-6
        gCost0 = (cost(hi0 + eps) - cost0) / eps
        if gCost0 < 0:
            hi1 = hi0 * explore_ratio
        else:
            hi1 = hi0 / explore_ratio
        cost1 = cost(hi1)
        s = (cost1 - cost0) / (hi1 - hi0)
        hi2 = (hi0 * s - 0.5 * gCost0 * (hi1 + hi0)) / (s - gCost0)
        cost2 = cost(hi2)
        if hi2 != None and hi2 > 0:
            if cost1 < cost0:
                if cost2 < cost1:
                    h = 1 / hi2
                else:
                    h = 1 / hi1
            else:
                if cost2 < cost0:
                    h = 1 / hi2
                else:
                    h = h0
        else:
            if cost1 < cost0:
                h = 1 / hi1
            else:
                h = h0
        return h



def naive_MMD_testsets_bandwidth(self, X, T0, T1, h0, metric_new):
    tau = 0.01
    hi0 = 1 / h0

    gmlpt_T0 = self.getGradientMinusLogPosterior_ensemble_new(T0)
    deltas_T0 = self.getDeltas(X, T0)
    metricDeltas_T0 = self.getMetricDeltas(metric_new, deltas_T0)
    deltaMetricDeltas_T0 = self.getDeltasMetricDeltas(deltas_T0, metricDeltas_T0)

    gmlpt_T1 = self.getGradientMinusLogPosterior_ensemble_new(T1)
    deltas_T1 = self.getDeltas(X, T1)
    metricDeltas_T1 = self.getMetricDeltas(metric_new, deltas_T1)
    deltaMetricDeltas_T1 = self.getDeltasMetricDeltas(deltas_T1, metricDeltas_T1)

    def cost(hi):
        h = 1 / hi
        try:
            assert(h > 0)
            kt0 = self.getKernelPiecewise(h, deltaMetricDeltas_T0)
            gkt0 = self.getGradKernelPiecewise(h, kt0, metricDeltas_T0)
            kt1 = self.getKernelPiecewise(h, deltaMetricDeltas_T1)
            gkt1 = self.getGradKernelPiecewise(h, kt1, metricDeltas_T1) # (Does this need a minus sign???) No, its in the method!
            T0_p = T0 + tau * self.mgJ_new(kt0, gkt0, gmlpt_T0)
            T1_p = T1 + tau * self.mgJ_new(kt1, gkt1, gmlpt_T1)
            return self.getMMD(T0_p, T1_p, mode='BM')
        except:
            log.info('Evaluating cost failed. Returning infty')
            return np.infty
    explore_ratio = 1.01 # used to be 1.1
    cost0 = cost(hi0)
    eps = 1e-6
    gCost0 = (cost(hi0 + eps) - cost0) / eps
    if gCost0 < 0:
        hi1 = hi0 * explore_ratio
    else:
        hi1 = hi0 / explore_ratio
    cost1 = cost(hi1)
    s = (cost1 - cost0) / (hi1 - hi0)
    hi2 = (hi0 * s - 0.5 * gCost0 * (hi1 + hi0)) / (s - gCost0)
    cost2 = cost(hi2)
    if hi2 != None and hi2 > 0:
        if cost1 < cost0:
            if cost2 < cost1:
                h = 1 / hi2
            else:
                h = 1 / hi1
        else:
            if cost2 < cost0:
                h = 1 / hi2
            else:
                h = h0
    else:
        if cost1 < cost0:
            h = 1 / hi1
        else:
            h = h0
    return h


# def test_dask(self):
#     bw = 1
#     f = h5py.File(self.ground_truth_path300)
#     d = f['/data']
#     y_ = da.from_array(d, chunks = 'auto')
#     dyy_ = self.getDeltas(y_, y_)
#     kyy_sum = np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dyy_, dyy_)))
#     pass

# def getMMD_new(self, X, Y, m=None):
#     """
#     Calculates MMD between X, Y. If Y is a ground truth provide path to disk instead
#     Args:
#         X: particle set (n x d array)
#         Y: particle set (n x d array) OR a path to a h5 file with an n x d array
#         m: number of samples in array from disk
#
#     Returns: MMD between X and Y (scalar)
#
#     """
#     Z = (2 * np.pi) ** (self.DoF / 2) # Normalizing factor
#     bw = 1 # bandwidth for kernel
#     n = X.shape[0]
#     dxx = self.getDeltas(X, X)
#     kxx_sum = np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dxx, dxx)))
#
#     if type(Y) == str:
#         G = 1 # number of gigs per chunk (uses two chunks)
#         chunk = np.sqrt(G * 1e9 / (8 * self.DoF)) # max chunk to work with in large array (n x n x d array)
#         # chunk = 500
#         assert(m != None)
#         kxy_sum = 0
#         kyy_sum = 0
#         tr = 0
#         num_chunks = int(np.floor(m / chunk))
#         num_remaining = m % chunk
#         # FOR TESTING PURPOSES
#         # dxy_total = self.getDeltas(X, self.loaded_array)
#         # dyy_total = self.getDeltas(self.loaded_array, self.loaded_array)
#         #
#         # dmdxy_total = self.getDeltasMetricDeltas(dxy_total, dxy_total)
#         # dmdyy_total = self.getDeltasMetricDeltas(dyy_total, dyy_total)
#         # kxy_total = self.getKernelPiecewise(1, dmdxy_total)
#         # kyy_total = self.getKernelPiecewise(1, dmdyy_total)
#         # kxy_sum_total = np.sum(kxy_total)
#         # kyy_sum_total = np.sum(kyy_total)
#         # pass
#
#         for i in range(num_chunks):
#             Y_ = dd.io.load(Y, sel=dd.aslice[i * chunk : (i+1) * chunk, :])
#             dxy = self.getDeltas(X, Y_)
#             dyy = self.getDeltas(Y_, Y_)
#             kxy = self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dxy, dxy)) #
#             kyy = self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dyy, dyy)) #
#             kxy_sum += np.sum(kxy)
#             kyy_sum += np.sum(np.triu(kyy))
#             tr += np.trace(kyy)
#             for j in range(num_chunks):
#                 if j > i:
#                     Y_1 = dd.io.load(Y, sel=dd.aslice[j * chunk : (j+1) * chunk, :])
#                     dyy = self.getDeltas(Y_, Y_1)
#                     kyy = self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dyy, dyy)) #
#                     kyy_sum += np.sum(kyy)
#
#         if num_remaining != 0:
#             Yr_ = dd.io.load(Y, sel = dd.aslice[-num_remaining:, :])
#             dxy = self.getDeltas(X, Yr_)
#             dyy = self.getDeltas(Yr_, Yr_)
#             kxy = self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD_excess_chunk(dxy, dxy))
#             kyy = self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD_excess_chunk(dyy, dyy))
#             kxy_sum += np.sum(kxy)
#             kyy_sum += np.sum(np.triu(kyy))
#             tr += np.trace(kyy)
#             for i in range(num_chunks):
#                 Y_ = dd.io.load(Y, sel=dd.aslice[i * chunk : (i+1) * chunk, :])
#                 dyy = self.getDeltas(Y_, Yr_)
#                 kyy = self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD_excess_chunk(dyy, dyy))
#                 kyy_sum += np.sum(kyy)
#
#         kyy_sum = 2 * kyy_sum - tr
#         return (1 / (n ** 2) * kxx_sum - 2 / (m * n) * kxy_sum + 1 / (m ** 2) * kyy_sum) / Z
#     else:
#         m = Y.shape[0]
#         dxy = self.getDeltas(X, Y)
#         dyy = self.getDeltas(Y, Y)
#         kxy_sum = np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dxy, dxy)))
#         kyy_sum = np.sum(self.getKernelPiecewise(bw, self.getDeltasMetricDeltas_MMD(dyy, dyy)))
#         return (1 / (n ** 2) * kxx_sum - 2 / (m * n) * kxy_sum + 1 / (m ** 2) * kyy_sum) / Z
# def getDeltasMetricDeltas_MMD_excess_chunk(self, deltas, metricDeltas):
#     """
#     FOR MMD
#     Requires n x d arrays for particles.
#     Args:
#         deltas:
#         metricDeltas:
#
#     Returns:
#
#     """
#     # TODO saves optimal contraction path. Comment this back in later. Did this to test MMD.
#     self.contract_d_md_MMD_excess_chunk = oe.contract_expression('mnd, mnd -> mn', deltas.shape, metricDeltas.shape)
#     return self.contract_d_md_MMD_excess_chunk(deltas, metricDeltas)


# def convergedMeanVar(self, Xl, Xlp1):
#
#     if self.accept_and_continue == True:
#         return False
#
#     xlmean = np.mean(Xl, axis=0)
#     xlp1mean = np.mean(Xlp1, axis=0)
#
#     xlvar = np.var(Xl, axis=0)
#     xlp1var = np.var(Xlp1, axis=0)
#
#     if self.mean_0_norm == None and self.variance_0_norm == None:
#         self.mean_0_norm, self.variance_0_norm = np.linalg.norm(xlmean), np.linalg.norm(xlvar)
#
#     a = 100 * np.linalg.norm(xlmean - xlp1mean) / self.mean_0_norm
#     b = 100 * np.linalg.norm(xlvar - xlp1var) / self.variance_0_norm
#
#     log.info('Mean Percent Change %f ::: Variance Percent Change %f' % (a, b))
#
#     tol_mean = 0.5
#     tol_var = 0.5
#     if a < tol_mean and b < tol_var:
#         return True
#     else:
#         return False





    def chooseSolverTol(self):
        self.atol = 1e-2
        # changes_alot = 20
        # changes_medium = 2
        #
        # self.extra_fine = 1e-3
        # self.fine = 1e-2
        # self.coarse = 1e-1
        # if self.phi_percent_change > changes_alot:
        #     self.atol = self.coarse
        # elif self.phi_percent_change < changes_alot and self.phi_percent_change > changes_medium:
        #     self.atol = self.fine
        # elif self.phi_percent_change < changes_medium and self.phi_percent_change > 0:
        #     self.atol = self.extra_fine
        # else:
        #     self.phi_increases = True

        # def getDeltasMetricDeltas_MMD_excess_chunk(self, deltas, metricDeltas):
    #     """
    #     FOR MMD
    #     Requires n x d arrays for particles.
    #     Args:
    #         deltas:
    #         metricDeltas:
    #
    #     Returns:
    #
    #     """
    #     # TODO saves optimal contraction path. Comment this back in later. Did this to test MMD.
    #     self.contract_d_md_MMD_excess_chunk = oe.contract_expression('mnd, mnd -> mn', deltas.shape, metricDeltas.shape)
    #     return self.contract_d_md_MMD_excess_chunk(deltas, metricDeltas)


    def getKernelWhole(self, bandwidth, metric, X, T_dict, skip = None):
        # skip tells you which key of the dict is the linesearch set, and skips.
        kernel_dict = {}
        # assert(len(T_list) >= 1)
        if self.iter_ == 0:
            delta_shape = (self.nParticles, self.nParticles, self.DoF)
            self.contract_dmd = oe.contract_expression('mnb, bd, mnd -> mn', delta_shape, metric.shape, delta_shape)
        if skip == None:
            for key in T_dict:
                T = T_dict[key]
                deltas = self.getDeltas(X, T)
                # kernel_dict[key] = np.exp(-1 / bandwidth * np.einsum('mnb, bd, mnd -> mn', deltas, metric, deltas))
                kernel_dict[key] = np.exp(-1 / bandwidth * self.contract_dmd(deltas, metric, deltas))
        else:
            for key in T_dict:
                T = T_dict[key]
                if key == skip:
                    kernel_dict[key] = None
                else:
                    deltas = self.getDeltas(X, T)
                    # kernel_dict[key] = np.exp(-1 / bandwidth * np.einsum('mnb, bd, mnd -> mn', deltas, metric, deltas))
                    kernel_dict[key] = np.exp(-1 / bandwidth * self.contract_dmd(deltas, metric, deltas))

        return kernel_dict



        def initializeTransportDict(self):
            transport_dict = dict()
        transport_dict['alphas'] = None
        transport_dict['bandwidth'] = None
        transport_dict['X'] = None
        transport_dict['metric'] = None
        transport_dict['eps'] = None
        return transport_dict

    def initializeMetadataDict(self):
        """
        Initializes dict for each iteration
        Returns:

        """
        metadata_dict = {}
        # metadata_dict['step_norm'] = None
        metadata_dict['method'] = None
        metadata_dict['MMD'] = None
        metadata_dict['nLikelihoodEvaluations'] = None
        metadata_dict['nGradLikelihoodEvaluations'] = None
        metadata_dict['nFisherLikelihoodEvaluations'] = None
        return metadata_dict

    def initializeTestSetDict(self):
        test_set_dict = dict()
        for t in range(self.nTestSets):
            test_set_dict['%i' % t] = None
        return test_set_dict

    def pickTestSet(self, iter_):
        index = iter_ % self.nTestSets
        if index == 0:
            self.a = np.random.choice(self.nTestSets, self.nTestSets, replace=False)
            if iter_ > 0:
                if self.a[0] == self.test_set_choice:
                    self.a = np.flip(self.a)
        return self.a[index]