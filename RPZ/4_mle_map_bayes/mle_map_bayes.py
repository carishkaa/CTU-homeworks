import numpy as np
from scipy.stats import norm
import scipy.special as spec  # for gamma


# MLE
def ml_estim_normal(x):
    """
    Computes maximum likelihood estimate of mean and variance of a normal distribution.

    :param x:   measurements (n, )
    :return:    mu - mean - python float
                var - variance - python float
    """
    n = np.shape(x)[0]
    mu = float(np.sum(x)/n)
    var = float(np.sum(np.square(x - mu))/n)
    return mu, var


def ml_estim_categorical(counts):
    """
    Computes maximum likelihood estimate of categorical distribution parameters.

    :param counts: measured bin counts (n, )
    :return:       pk - (n, ) parameters of the categorical distribution
    """
    pk = counts/np.sum(counts)
    return pk


# MAP
def map_estim_normal(x, mu0, nu, alpha, beta):
    """
    Maximum a posteriori parameter estimation of normal distribution with normal inverse gamma prior.

    :param x:      measurements (n, )
    :param mu0:    NIG parameter - python float
    :param nu:     NIG parameter - python float
    :param alpha:  NIG parameter - python float
    :param beta:   NIG parameter - python float

    :return:       mu - estimated mean - python float
    :return:       var - estimated variance - python float
    """
    n = np.shape(x)[0]
    mu = float((nu * mu0 + np.sum(x))/(n + nu))
    var = float((2 * beta + nu * np.square(mu0 - mu) + np.sum(np.square(x - mu)))/(n + 3 + 2 * alpha))
    return mu, var


def map_estim_categorical(counts, alpha):
    """
    Maximum a posteriori parameter estimation of categorical distribution with Dirichlet prior.

    :param counts:  measured bin counts (n, )
    :param alpha:   Dirichlet distribution parameters (n, )

    :return:        pk - estimated categorical distribution parameters (n, )
    """
    pk = (counts + alpha - 1)/np.sum(counts + alpha - 1)
    return pk


# BAYES
def bayes_posterior_params_normal(x, prior_mu0, prior_nu, prior_alpha, prior_beta):
    """
    Compute a posteriori normal inverse gamma parameters from data and NIG prior.

    :param x:            measurements (n, )
    :param prior_mu0:    NIG parameter - python float
    :param prior_nu:     NIG parameter - python float
    :param prior_alpha:  NIG parameter - python float
    :param prior_beta:   NIG parameter - python float

    :return:             mu0:    a posteriori NIG parameter - python float
    :return:             nu:     a posteriori NIG parameter - python float
    :return:             alpha:  a posteriori NIG parameter - python float
    :return:             beta:   a posteriori NIG parameter - python float
    """
    n = np.shape(x)[0]
    alpha = float(prior_alpha + n/2)
    nu = float(prior_nu + n)
    mu0 = float((prior_nu * prior_mu0 + np.sum(x)) / (prior_nu + n))
    beta = float(prior_beta + 0.5 * np.sum(x*x) + 0.5 * prior_nu * np.square(prior_mu0) - (0.5 * np.square(prior_nu * prior_mu0 + np.sum(x)))/(prior_nu + n))
    return mu0, nu, alpha, beta


def bayes_posterior_params_categorical(counts, alphas):
    """
    Compute a posteriori Dirichlet parameters from data and Dirichlet prior.

    :param counts:   measured bin counts (n, )
    :param alphas:   prior Dirichlet distribution parameters (n, )

    :return:         posterior_alphas - estimated Dirichlet distribution parameters (n, )
    """
    posterior_alphas = counts + alphas
    return posterior_alphas


def bayes_estim_pdf_normal(x_test, x, mu0, nu, alpha, beta):
    """
    Compute pdf of predictive distribution for Bayesian estimate for normal distribution with normal inverse gamma prior.

    :param x_test:  values where the pdf should be evaluated (m, )
    :param x:       'training' measurements (n, )
    :param mu0:     prior NIG parameter - python float
    :param nu:      prior NIG parameter - python float
    :param alpha:   prior NIG parameter - python float
    :param beta:    prior NIG parameter - python float

    :return:        pdf - Bayesian estimate pdf evaluated at x_test (m, )
    """
    mu0, nu, alpha, beta = bayes_posterior_params_normal(x, mu0, nu, alpha, beta)

    alpha_k = alpha + 0.5
    nu_k = nu + 1
    pdf = np.array([])
    for x_i in x_test:
        beta_k = np.square(x_i)/2 + beta + (nu * np.square(mu0))/2 - np.square(nu*mu0 + x_i)/(2 * (nu + 1))
        pdf = np.append(pdf, 1 / np.sqrt(2 * np.pi) * (np.sqrt(nu) * np.power(beta, alpha))/(np.sqrt(nu_k) * np.power(beta_k, alpha_k)) * spec.gamma(alpha_k)/spec.gamma(alpha))
    return pdf


def bayes_estim_categorical(counts, alphas):
    """
    Compute parameters of Bayesian estimate for categorical distribution with Dirichlet prior.

    :param counts:  measured bin counts (n, )
    :param alphas:  prior Dirichlet distribution parameters (n, )

    :return:        pk - estimated categorical distribution parameters (n, )
    """
    pk = (counts + alphas)/np.sum(counts + alphas)
    return pk


# Classification
def mle_Bayes_classif(test_imgs, train_data_A, train_data_C):
    """
    Classify images using Bayes classification using MLE of normal distributions and 0-1 loss.

    :param test_imgs:      images to be classified (H, W, N)
    :param train_data_A:   training image features A (nA, )
    :param train_data_C:   training image features C (nC, )

    :return:               q - classification strategy (see find_strategy_2normal)
    :return:               labels - classification of test_imgs (N, ) (see bayes.classify_2normal)
    :return:               DA - parameters of the normal distribution of A
                            DA['Mean'] - python float
                            DA['Sigma'] - python float
                            DA['Prior'] - python float
    :return:               DC - parameters of the normal distribution of C
                            DC['Mean'] - python float
                            DC['Sigma'] - python float
                            DC['Prior'] - python float
    """
    DA = dict()
    DC = dict()
    DA['Mean'], DA['Sigma'] = ml_estim_normal(train_data_A)
    DC['Mean'], DC['Sigma'] = ml_estim_normal(train_data_C)
    DA['Sigma'] = float(np.sqrt(DA['Sigma']))
    DC['Sigma'] = float(np.sqrt(DC['Sigma']))
    nA = np.shape(train_data_A)[0]
    nC = np.shape(train_data_C)[0]
    DA['Prior'] = nA / (nA + nC)
    DC['Prior'] = nC / (nA + nC)

    q = find_strategy_2normal(DA, DC)
    labels = classify_2normal(test_imgs, q)

    return q, labels, DA, DC


def map_Bayes_classif(test_imgs, train_data_A, train_data_C,
                      mu0_A, nu_A, alpha_A, beta_A,
                      mu0_C, nu_C, alpha_C, beta_C):
    """
    Classify images using Bayes classification using MAP estimate of normal distributions with NIG priors and 0-1 loss.

    :param test_imgs:      images to be classified (H, W, N)
    :param train_data_A:   training image features A (nA, )
    :param train_data_C:   training image features C (nC, )

    :param mu0_A:          prior NIG parameter for A - python float
    :param nu_A:           prior NIG parameter for A - python float
    :param alpha_A:        prior NIG parameter for A - python float
    :param beta_A:         prior NIG parameter for A - python float

    :param mu0_C:          prior NIG parameter for C - python float
    :param nu_C:           prior NIG parameter for C - python float
    :param alpha_C:        prior NIG parameter for C - python float
    :param beta_C:         prior NIG parameter for C - python float

    :return:               q - classification strategy (see find_strategy_2normal)
    :return:               labels - classification of test_imgs (N, ) (see bayes.classify_2normal)
    :return:               DA - parameters of the normal distribution of A
                            DA['Mean'] - python float
                            DA['Sigma'] - python float
                            DA['Prior'] - python float
    :return:               DC - parameters of the normal distribution of C
                            DC['Mean'] - python float
                            DC['Sigma'] - python float
                            DC['Prior'] - python float
    """
    DA = dict()
    DC = dict()
    DA['Mean'], DA['Sigma'] = map_estim_normal(train_data_A, mu0_A, nu_A, alpha_A, beta_A)
    DC['Mean'], DC['Sigma'] = map_estim_normal(train_data_C, mu0_C, nu_C, alpha_C, beta_C)
    DA['Sigma'] = float(np.sqrt(DA['Sigma']))
    DC['Sigma'] = float(np.sqrt(DC['Sigma']))
    nA = np.shape(train_data_A)[0]
    nC = np.shape(train_data_C)[0]
    DA['Prior'] = nA / (nA + nC)
    DC['Prior'] = nC / (nA + nC)
    q = find_strategy_2normal(DA, DC)
    labels = classify_2normal(test_imgs, q)
    return q, labels, DA, DC


def bayes_Bayes_classif(x_test, x_train_A, x_train_C,
                        mu0_A, nu_A, alpha_A, beta_A,
                        mu0_C, nu_C, alpha_C, beta_C):
    """
    Classify images using Bayes classification (0-1 loss) using predictive pdf estimated using Bayesian inferece with NIG priors.

    :param x_test:         images features to be classified (n, )
    :param x_train_A:      training image features A (nA, )
    :param x_train_C:      training image features C (nC, )

    :param mu0_A:          prior NIG parameter for A - python float
    :param nu_A:           prior NIG parameter for A - python float
    :param alpha_A:        prior NIG parameter for A - python float
    :param beta_A:         prior NIG parameter for A - python float

    :param mu0_C:          prior NIG parameter for C - python float
    :param nu_C:           prior NIG parameter for C - python float
    :param alpha_C:        prior NIG parameter for C - python float
    :param beta_C:         prior NIG parameter for C - python float

    :return:               labels - classification of x_test (n, ) int32, values 0 or 1
    """
    nA = np.shape(x_train_A)[0]
    nC = np.shape(x_train_C)[0]
    p_A = bayes_estim_pdf_normal(x_test, x_train_A, mu0_A, nu_A, alpha_A, beta_A)
    p_C = bayes_estim_pdf_normal(x_test, x_train_C, mu0_C, nu_C, alpha_C, beta_C)
    apriori_A = nA / (nA + nC)
    apriori_C = nC / (nA + nC)

    labels = np.where(p_A * apriori_A > p_C * apriori_C, 0, 1)
    return np.int32(labels)


#### Previous labs here:

def find_strategy_2normal(distribution_A, distribution_B):
    """
    q = find_strategy_2normal(distribution_A, distribution_B)

    Find optimal bayesian strategy for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] - python floats
    :param distribution_B:  the same as distribution_A

    :return q:              strategy dict
                               q['t1'], q['t2'] - decision thresholds - python floats
                               q['decision'] - (3, ) np.int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
                               If there is only one threshold, q['t1'] should be equal to q['t2'] and the middle decision should be 0
                               If there is no threshold, q['t1'] and q['t2'] should be -/+ infinity and all the decision values should be the same (0 preferred)
    """
    q = dict()
    q['t1'] = -float('inf')
    q['t2'] = float('inf')

    if distribution_A['Prior'] == 1:
        q['decision'] = np.int32(np.array([0, 0, 0]))
        return q

    if distribution_B['Prior'] == 1:
        q['decision'] = np.int32(np.array([1, 1, 1]))
        return q

    sigma_A = distribution_A['Sigma']
    sigma_B = distribution_B['Sigma']
    mean_A = distribution_A['Mean']
    mean_B = distribution_B['Mean']
    L = np.log(sigma_A * distribution_B['Prior']/(sigma_B * distribution_A['Prior']))

    # coefficients
    a = np.square(sigma_A) - np.square(sigma_B)
    b = 2 * (mean_A * np.square(sigma_B) - mean_B * np.square(sigma_A))
    c = np.square(mean_B) * np.square(sigma_A) - \
        np.square(mean_A) * np.square(sigma_B) - \
        2 * L * np.square(sigma_A) * np.square(sigma_B)

    # compute roots
    roots = np.roots([a, b, c])

    # make decision
    if mean_A == mean_B and sigma_A == sigma_B:  # no roots (a = 0, b = 0)
        q['decision'] = np.int32(np.array([0, 0, 0])) if c >= 0 else np.int32(np.array([1, 1, 1]))
        return q

    if sigma_A == sigma_B:  # linear (a = 0)
        q['t1'] = q['t2'] = float(roots[0])
        q['decision'] = np.int32(np.array([1, 0, 0])) if b > 0 else np.int32(np.array([0, 0, 1]))
        return q

    if not np.logical_and.reduce(np.isreal(roots)):  # if there are complex roots (high prior + high variance)
        q['decision'] = np.int32(np.array([0, 0, 0])) if a >= 0 else np.int32(np.array([1, 1, 1]))
        return q

    if roots[0] == roots[1]:   # same roots
        q['t1'] = q['t2'] = float(roots[0])
        q['decision'] = np.int32(np.array([0, 0, 0])) if a > 0 else np.int32(np.array([1, 0, 1]))
        return q
    else:
        q['t1'] = float(min(roots))
        q['t2'] = float(max(roots))
        q['decision'] = np.int32(np.array([0, 1, 0])) if a > 0 else np.int32(np.array([1, 0, 1]))
        return q


def classify_2normal(imgs, q):
    """
    label = classify_2normal(imgs, q)

    Classify images using continuous measurement and strategy q.

    :param imgs:    test set images, np.array (h, w, n)
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) int32 np.array decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        label - image labels, (n, ) int32
    """
    x = compute_measurement_lr_cont(imgs)
    label = np.array([])
    for i in x:
        if i <= q['t1']:
            label = np.append(label, q['decision'][0])
        if q['t1'] < i <= q['t2']:
            label = np.append(label, q['decision'][1])
        if i > q['t2']:
            label = np.append(label, q['decision'][2])
    return np.int32(label)


def compute_measurement_lr_cont(imgs):
    """
    x = compute_measurement_lr_cont(imgs)

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:    set of images, (h, w, n)
    :return:        measurements, (n, )
    """
    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2),:], axis=0) - np.sum(sum_rows[int(width / 2):,:], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def classification_error_2normal(images, labels, q):
    """
    error = classification_error_2normal(images, labels, q)

    Compute classification error of a strategy q in a test set.

    :param images:  test set images, (h, w, n)
    :param labels:  test set labels (n, )
    :param q:       strategy
                    q['t1'] q['t2'] - float decision thresholds
                    q['decision'] - (3, ) np.int32 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:        python float classification error in range <0, 1>. Fraction of incorrect classifications.
    """
    classified_labels = classify_2normal(images, q)
    error = float(np.sum(classified_labels != labels) / np.shape(labels)[0])
    return error

#### provided functions

def mle_likelihood_normal(x, mu, var):
    """
    Compute the likelihood of the data x given the model is a normal distribution with given mean and sigma

    :param x:       measurements (n, )
    :param mu:      the normal distribution mean
    :param var:     the normal distribution variance
    :return:        L - likelihood of the data x
    """
    assert len(x.shape) == 1

    if var <= 0:
        L = 0
    else:
        L = np.prod(norm.pdf(x, mu, np.sqrt(var)))
    return L


def norm_inv_gamma_pdf(mu, var, mu0, nu, alpha, beta):
    # Wikipedia sometimes uses a symbol 'lambda' instead 'nu'

    assert alpha > 0
    assert nu > 0
    if beta <= 0 or var <= 0:
        return 0

    sigma = np.sqrt(var)

    p = np.sqrt(nu) / (sigma * np.sqrt(2 * np.pi)) * np.power(beta, alpha) / spec.gamma(alpha) * np.power(1/var, alpha + 1) * np.exp(-(2 * beta + nu * (mu0 - mu) * (mu0 - mu)) / (2 * var))

    return p
