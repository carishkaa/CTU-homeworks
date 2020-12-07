import matplotlib.pyplot as plt
import numpy as np
import mle_map_bayes as mmb

# plots for Normal distribution
def plot_likelihood(x, mu_min, mu_max, var_min, var_max, do_plot=True):
    mu_grid, var_grid = np.meshgrid(np.linspace(mu_min, mu_max, 100), np.linspace(var_min, var_max, 100))
    L_grid = likelihood_normal_grid(x, mu_grid, var_grid)
    if do_plot:
        plt.imshow(L_grid, interpolation='bilinear', origin='lower', extent=[mu_min, mu_max, var_min, var_max],
                   aspect='auto')
        plt.xlabel('mu')
        plt.ylabel('var')
        plt.title('p(x | mu, var)')
    return L_grid


def plot_prior(mu0, nu, alpha, beta, mu_min, mu_max, var_min, var_max, plot=True):
    mu_grid, var_grid = np.meshgrid(np.linspace(mu_min, mu_max, 100), np.linspace(var_min, var_max, 100))
    prior_grid = norm_inv_gamma_grid(mu_grid, var_grid, mu0, nu, alpha, beta)
    if plot:
        plt.imshow(prior_grid, interpolation='bilinear', origin='lower', extent=[mu_min, mu_max, var_min, var_max],
                   aspect='auto')
        plt.xlabel('mu')
        plt.ylabel('var')
        plt.title('p(mu, var)')
    return prior_grid


def plot_MAP_objective(x, mu0, nu, alpha, beta, mu_min, mu_max, var_min, var_max):
    mu_grid, var_grid = np.meshgrid(np.linspace(mu_min, mu_max, 100), np.linspace(var_min, var_max, 100))

    L_grid = likelihood_normal_grid(x, mu_grid, var_grid)
    prior_grid = norm_inv_gamma_grid(mu_grid, var_grid, mu0, nu, alpha, beta)

    # The following plot is NOT a probability distribution!!! It is not normalised to sum up to one!
    plt.imshow(np.multiply(L_grid, prior_grid), interpolation='bilinear', origin='lower',
               extent=[mu_min, mu_max, var_min, var_max], aspect='auto')
    plt.xlabel('mu')
    plt.ylabel('var')
    plt.title('MAP objective')


def plot_posterior_normal(x, mu0, nu, alpha, beta, mu_min, mu_max, var_min, var_max, plot=True):
    mu_grid, var_grid = np.meshgrid(np.linspace(mu_min, mu_max, 100), np.linspace(var_min, var_max, 100))
    grid = posterior_grid(mu_grid, var_grid, x, mu0, nu, alpha, beta)
    if plot:
        plt.imshow(grid, interpolation='bilinear', origin='lower', extent=[mu_min, mu_max, var_min, var_max],
                   aspect='auto')
        plt.xlabel('mu')
        plt.ylabel('var')
        plt.title('p(mu, var | x)')
    return grid


# plots for categorical distribution
def plot_categorical_distr(pc, title):
    num_classes = len(pc)
    plt.bar(np.arange(num_classes) + 1, pc)
    for i, v in enumerate(pc):
        plt.text(i + 1, v + 0.01, '{:.2f}'.format(v), color='black', ha='center')
    plt.xlabel('class')
    plt.ylabel('p(class)')
    plt.title(title)


def plot_categorical_hist(counts, title):
    num_classes = len(counts)
    plt.bar(np.arange(num_classes) + 1, counts)
    for i, v in enumerate(counts):
        plt.text(i + 1, v + 0.5, '{:d}'.format(v), color='black', ha='center')
    plt.xlabel('class')
    plt.ylabel('# samples')
    plt.title(title)


# helper functions
def likelihood_normal_grid(x, mu_grid, var_grid):
    L = np.zeros_like(mu_grid)
    M, N = mu_grid.shape
    for m in range(M):
        for n in range(N):
            L[m, n] = mmb.mle_likelihood_normal(x, mu_grid[m, n], var_grid[m, n])

    return L


def norm_inv_gamma_grid(mu_grid, var_grid, mu0, nu, alpha, beta):
    L = np.zeros_like(mu_grid)
    M, N = mu_grid.shape
    for m in range(M):
        for n in range(N):
            L[m, n] = mmb.norm_inv_gamma_pdf(mu_grid[m, n], var_grid[m, n], mu0, nu, alpha, beta)

    return L


def posterior_grid(mu_grid, var_grid, x, mu0, nu, alpha, beta):
    L = np.zeros_like(mu_grid)

    mu0_, nu_, alpha_, beta_ = mmb.bayes_posterior_params_normal(x, mu0, nu, alpha, beta)
    # print('alpha_ = {:.2f}, beta_ = {:.2f}, gamma_ = {:.2f}, delta_ = {:.2f}'.format(alpha_, beta_, gamma_, delta_))
    M, N = mu_grid.shape
    for m in range(M):
        for n in range(N):
            L[m, n] = mmb.norm_inv_gamma_pdf(mu_grid[m, n], var_grid[m, n], mu0_, nu_, alpha_, beta_)

    return L
