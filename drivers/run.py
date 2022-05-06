############################################################################
# Import posterior models
############################################################################
# from models.gauss_stats import gauss
from models.hybrid_rosenbrock import hybrid_rosenbrock
# from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
###########################################################################
# Import SVI suite
###########################################################################
from src.samplers import samplers
###########################################################################
# Import custom methods
###########################################################################
from scripts.create_contour import create_contour
from scripts.create_animation import animate_driver
###########################################################################
# Import useful plotting functions
###########################################################################
from plots.plot_final_samples_static import make_final_samples_plot
# from plots.moment_convergence_plot import plot_moment_convergence
from plots.moment_convergence_plot_relative import plot_moment_convergence
from plots.cornerplots import make_corner_plots
# from plots.cornerplots_corner import make_corner_plots
# from plots.cornerplots import make_corner_plots
from plots.particle_trajectory_plot import create_particle_trajectory_plots
from scripts.pp_plots import make_pp_plots
###########################################################################
# Import libraries
###########################################################################
import argparse
import logging.config
import os
import corner
import h5py
import numpy as np

logger = logging.getLogger(__name__)
def main():
    #########################################
    # Settings for 2D-Gaussian
    #########################################
    # model = gauss_stats(2)
    ###############################################
    # Settings for 15D-HRD (hard to run on laptop)
    ###############################################
    # n2 = 7
    # n1 = 3
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)
    #########################################
    # Settings for 10D-HRD
    #########################################
    n2 = 3
    n1 = 4
    model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20)
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=10, b=np.ones((n2, n1-1)) * 30)
    #########################################
    # Settings for 5D-HRD
    #########################################
    # n2 = 2
    # n1 = 3
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20, id='easier)
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=10, b=np.ones((n2, n1-1)) * 30, id='harder') # harder and better one
    #########################################
    # Settings for 2D-HRD (Gaussian like)
    #########################################
    # n2 = 1
    # n1 = 2
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20, id='gaussian-like')
    ##########################################
    # Settings for 2D-HRD (Thin like)
    ########################################
    # n2 = 1
    # n1 = 2
    # model = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5, id='thin-like')
    #########################################
    # Double banana
    #########################################
    # model = rosenbrock_analytic()
    ####################################################################################
    # Perform desired inference
    ####################################################################################
    stein = samplers(model=model, nIterations=args.nIterations, nParticles=args.nParticles, profile=args.profile)
    # stein.deterministic('SVN/SVGD', 'LS/CONSTANT', 'MED/HE/BM/CONSTANT')
    # stein.stochastic('SVN/SVGD')
    # output_dict = stein.deterministic(method=args.optimize_method, eps=1, h=40)
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.5, h=20) # Good for sSVNa 10D
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.1, h=.4) #for SVGD
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.1, h=4) #for SVGD
    # output_dict = stein.stochastic(method=args.optimize_method, eps=.5, h=40)
    # This works well
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.5, h=20) # testing for 10D
    # output_dict = stein.stochastic(method=args.optimize_method, eps=1, h=20) # Works well for KHK
    # output_dict = stein.stochastic(method=args.optimize_method, eps=.1) # Works well for KHK
    output_dict = stein.apply(method=args.optimize_method, eps=0.1)
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.5, h=40)
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.5, h=10)
    # output_dict = stein.stochastic(method=args.optimize_method, eps=0.5, h=10) # 1 works well for 2d

    run_output_path = output_dict['outdir_path']
    file = output_dict['path_to_results']
    dict1 = {'%s' % args.optimize_method : file}
    try:
        GT = model.newDrawFromLikelihood(2000000)
    except:
        GT = None

    # Store figures '/figures' folder
    figures_directory = os.path.join(run_output_path, 'figures')
    if os.path.isdir(figures_directory) is False:
        os.mkdir(figures_directory)

    if model.DoF == 2:
        try: # Create contour plot
            contour_file_path = create_contour(stein, stein.OUTPUT_DIR)
        except Exception as e: log.error(e)

        try:
            log.info('Making final samples plot:')
            make_final_samples_plot(dict1, contour_file_path, GT, save_path=figures_directory)
        except Exception as e: log.error(e)
    ###################################################################
    # Moment convergence
    ###################################################################
    try:
        log.info('Making moment convergence plots:')
        plot_moment_convergence(dict1, GT=GT, save_path=figures_directory)
    except Exception as e: log.error(e)
    ###################################################################
    # Corner plots
    ###################################################################
    try:
        log.info('Making corner plots:')
        make_corner_plots(dict1, GT, save_path=figures_directory)
    except Exception as e: log.error(e)

    try:
        log.info('Making pp-plots:')
        # dict1 = {'%s' % args.optimize_method : file}
        make_pp_plots(dict1, GT, save_path=figures_directory)
    except Exception as e: log.error(e)

    if model.DoF == 2:
        try: # Draw particle history atop contour plot
            log.info('Making particle trajectory plots')
            create_particle_trajectory_plots(run_output_path, contour_file_path)
        except Exception as e: log.error(e)

        try: # Draw particle history atop contour plot
            log.info('Making animation')
            animate_driver(contour_file_path, run_output_path)
        except Exception as e: log.error(e)










    # try:
    #     with h5py.File(history_path, 'r') as f:
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
    #         # final_particles = f['final_updated_particles']['X'][:]
    #     # fig = corner.corner(particles)
    #     fig = corner.corner(GT)
    #     filename = os.path.join(run_output_path, 'figures', 'corner.png')
    #     # filename = run_output_path + 'figures/' + 'corner.png'
    #     fig.savefig(filename)
    # except:
    #     log.error('Failed to make corner plots')

    # try:
    #     log.info('Making MMD plots...')
    #     if model.modelType == 'gauss_analytic':
    #         target_samples = np.random.multivariate_normal(stein.model.mu_l, stein.model.sigma_l, 2000)
    #     create_MMD_plots(run_output_path, target_samples)
    # except:
    #     log.info('Failed to make MMD')
    #     pass

    # Animate driver makes the figures folder at the moment!!! Make sure at least one runs or other figures wont find directory!
    # if stein.model.DoF == 2:
    #     try:
    #         log.info('Animating basis')
    #         animate_driver(history_path, contour_file_path, RUN_OUTPUT_DIR=run_output_path)#composite_map=composite_map)
    #     except Exception as e: log.info(e)
        # try:
        #     for m in range(stein.nTestSets):
        #         log.info('Animating test set %i' % m)
        #         animate_driver(contour_file_path=contour_file_path, output_dir=run_output_path, test_set_dict=test_set_history, m=m, method=method)#composite_map=composite_map)
        # except:
        #     log.info('Failed to animate test sets')

    # log.info('Creating displacement field plots')
    # dfp = displacement_field_plots(stein, run_output_path)
    # dfp.get_pushforward_info(dfp.grid_particles)
    # dfp.animate_displacement_fields()
    #
    # log.info('Creating kdes')
    # create_kde_plots(composite_map, run_output_path, stein.model.begin, stein.model.end)

def load_logger(config_name):
    """
    Load configuration file for logger from same folder as driver
    Args:
        file_name: .ini file name

    Returns: log object

    """
    directory = os.path.dirname(os.path.abspath(__file__))
    logger_config_path = '%s/%s' % (directory, config_name)
    assert os.path.exists(logger_config_path), 'Logger configuration file not found.'
    logging.config.fileConfig(logger_config_path, disable_existing_loggers=False)
    return logging.getLogger()

if __name__ == '__main__':
    # Setup argparse
    parser = argparse.ArgumentParser(description='Driver for Stein Variational Inference Package')
    parser.add_argument('-nIter', '--nIterations', type=int, required=True, help='Number of iterations')
    parser.add_argument('-nP', '--nParticles', type=int, required=True, help='Number of particles')
    parser.add_argument('-opt', '--optimize_method', type=str, required=True, help='Choice of optimization')
    parser.add_argument('-prf', '--profile', type=str, required=False, help='Output algorithm profile information')
    args = parser.parse_args()

    config_name = 'logger_configuration.ini'
    log = load_logger(config_name)

    log.info('Beginning job')
    main()
    log.info('Ending job')


