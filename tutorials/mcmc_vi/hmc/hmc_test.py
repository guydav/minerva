import hamiltonian_monte_carlo as hmc
import diagnostics
import pickle
import numpy as np


def two_d_mixture_gaussians(theta):
    return hmc.np.log(0.5 * hmc.stats.multivariate_normal.pdf(theta, mean=hmc.np.array([2, 12]),
                                                          cov=hmc.np.array(([[2, 0.5], [0.5, 1]]))) + \
                  0.25 * hmc.stats.multivariate_normal.pdf(theta, mean=hmc.np.array([10, -8]),
                                                           cov=hmc.np.array(([[1, 0.9], [0.9, 2]]))) + \
                  0.25 * hmc.stats.multivariate_normal.pdf(theta, mean=hmc.np.array([-12, -9]),
                                                           cov=hmc.np.array(([[1, 0], [0, 1]]))))


if __name__ == '__main__':
    # s = hmc.hamiltonian_monte_carlo(two_d_mixture_gaussians, [1.0, 1.0], 0.05, 30, 30, n_var=2)
    # print(s)

    # with open('results.pickle', 'rb') as f:
    #     results = pickle.load(f)
    #
    # all_samples, all_theta_samples = results[(0.8, 400)]
    # theta_samples = np.asarray(all_theta_samples)[:, 2000:, :]
    # ess = diagnostics.effective_sample_size(theta_samples)
    # rhat = diagnostics.split_r_hat(theta_samples)

    theta_0_sampler = lambda: np.random.uniform(-10, 10, 2)

    epsilon = 0.1
    momentum_std = 1.0

    samples = hmc.no_u_turn_sampler(two_d_mixture_gaussians, theta_0_sampler(),
                                    epsilon, 2000, 2, momentum_std=momentum_std,
                                    log_interval=100)