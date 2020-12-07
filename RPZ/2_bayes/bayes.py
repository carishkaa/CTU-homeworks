#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.stats import norm
import math
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from PIL import Image


def bayes_risk_discrete(discrete_A, discrete_B, W, q):
    """
    R = bayes_risk_discrete(discrete_A, discrete_B, W, q)

    Compute bayesian risk for a discrete strategy q

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param W:                       cost function np.array (states, decisions)
                                    (nr. of states and decisions is fixed to 2)
    :param q:                       strategy - (n, ) np.array, values 0 or 1
    :return:                        bayesian risk - python float
    """
    n = np.shape(q)[0]  # 21
    joint_prob_A = discrete_A['Prior'] * discrete_A['Prob']
    joint_prob_B = discrete_B['Prior'] * discrete_B['Prob']
    X_set = np.arange(n)
    prob = [(joint_prob_A[x] * W[0, q[x]] + joint_prob_B[x] * W[1, q[x]]) for x in X_set]
    R = float(np.sum(prob))
    return R


def find_strategy_discrete(discrete_A, discrete_B, W):
    """
    q = find_strategy_discrete(distribution1, distribution2, W)

    Find bayesian strategy for 2 discrete distributions.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param W:                       cost function np.array (states, decisions)
                                    (nr. of states and decisions is fixed to 2)
    :return:                        q - optimal strategy (n, ) np.array, values 0 or 1
    """
    joint_prob_A = discrete_A['Prior'] * discrete_A['Prob']
    joint_prob_B = discrete_B['Prior'] * discrete_B['Prob']
    prob = [joint_prob_A * W[0, 0] + joint_prob_B * W[1, 0],
            joint_prob_A * W[0, 1] + joint_prob_B * W[1, 1]]
    q = np.argmin(prob, axis=0)
    return q


def classify_discrete(imgs, q):
    """
    function label = classify_discrete(imgs, q)

    Classify images using discrete measurement and strategy q.

    :param imgs:    test set images, (h, w, n) uint8 np.array
    :param q:       strategy (21, ) np.array of 0 or 1
    :return:        image labels, (n, ) np.array of 0 or 1
    """
    x = compute_measurement_lr_discrete(imgs).astype(int)
    label = np.array([q[i + 10] for i in x])  # x = [-10, 10]
    return label


def classification_error_discrete(images, labels, q):
    """
    error = classification_error_discrete(images, labels, q)

    Compute classification error for a discrete strategy q.

    :param images:      images, (h, w, n) np.uint8 np.array
    :param labels:      (n, ) np.array of values 0 or 1 - ground truth labels
    :param q:           (m, ) np.array of 0 or 1 - classification strategy
    :return:            error - classification error as a fraction of false samples
                        python float in range <0, 1>
    """
    classified_labels = classify_discrete(images, q)
    error = float(np.sum(classified_labels != labels) / np.shape(labels)[0])
    return error


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


def bayes_risk_2normal(distribution_A, distribution_B, q):
    """
    R = bayes_risk_2normal(distribution_A, distribution_B, q)

    Compute bayesian risk of a strategy q for 2 normal distributions and zero-one loss function.

    :param distribution_A:  parameters of the normal dist.
                            distribution_A['Mean'], distribution_A['Sigma'], distribution_A['Prior'] python floats
    :param distribution_B:  the same as distribution_A
    :param q:               strategy
                               q['t1'], q['t2'] - float decision thresholds (python floats)
                               q['decision'] - (3, ) np.int32 np.array 0/1 decisions for intervals (-inf, t1>, (t1, t2>, (t2, inf)
    :return:    R - bayesian risk, python float
    """

    intervals = [[-float('inf'), q['t1']],
                 [q['t1'], q['t2']],
                 [q['t2'], float('inf')]]
    sigma = [distribution_A['Sigma'], distribution_B['Sigma']]
    mean = [distribution_A['Mean'], distribution_B['Mean']]
    prior = [distribution_A['Prior'], distribution_B['Prior']]

    q1, q2, q3 = q['decision']

    f = [norm.cdf(intervals[0], mean[q1], sigma[q1]),
         norm.cdf(intervals[1], mean[q2], sigma[q2]),
         norm.cdf(intervals[2], mean[q3], sigma[q3])]

    integrals = [f[0][1] - f[0][0],
                 f[1][1] - f[1][0],
                 f[2][1] - f[2][0]]

    R = float(1 - (prior[q1] * integrals[0] + prior[q2] * integrals[1] + prior[q3] * integrals[2]))
    return R


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


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################


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


def compute_measurement_ul_cont(imgs):
    """
    x = compute_measurement_ul_cont(imgs)

    Compute measurement on images, subtract sum of upper half from sum of
    lower half.

    :param imgs:    set of images, (h, w, n)
    :return:        measurements, (n, )
    """
    assert len(imgs.shape) == 3

    h = imgs.shape[0]
    sum_columns = np.sum(imgs, dtype=np.float64, axis=1)

    x = np.sum(sum_columns[0:int(h / 2), :], axis=0) - np.sum(sum_columns[int(h / 2):, :], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x


def visualize_discrete(discrete_A, discrete_B, q):
    """
    visualize_discrete(discrete_A, discrete_B, q)

    Visualize a strategy for 2 discrete distributions.

    :param discrete_A['Prob']:      pXk(x|A) given as a (n, ) np.array
    :param discrete_A['Prior']:     prior probability pK(A)
    :param discrete_B['Prob']:      pXk(x|B) given as a (n, ) np.array
    :param discrete_B['Prior']:     prior probability pK(B)
    :param q:                       strategy - (n, ) np.array, values 0 or 1
    """

    posterior_A = discrete_A['Prob'] * discrete_A['Prior']
    posterior_B = discrete_B['Prob'] * discrete_B['Prior']

    max_prob = np.max([posterior_A, posterior_B])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Posterior probabilities and strategy q")
    plt.xlabel("feature")
    plt.ylabel("posterior probabilities")

    bins = np.array(range(posterior_A.size + 1)) - int(posterior_A.size / 2)

    width = 0.75
    bar_plot_A = plt.bar(bins[:-1], posterior_A, width=width, color='b', alpha=0.75)
    bar_plot_B = plt.bar(bins[:-1], posterior_B, width=width, color='r', alpha=0.75)

    plt.legend((bar_plot_A, bar_plot_B), (r'$p_{XK}(x,A)$', r'$p_{XK}(x,B)$'))

    sub_level = - max_prob / 8
    height = np.abs(sub_level)
    for idx in range(len(bins[:-1])):
        b = bins[idx]
        col = 'r' if q[idx] == 1 else 'b'
        patch = patches.Rectangle([b - 0.5, sub_level], 1, height, angle=0.0, color=col, alpha=0.75)
        ax.add_patch(patch)

    plt.ylim(bottom=sub_level)
    plt.text(bins[0], -max_prob / 16, 'strategy q')


def visualize_2norm(cont_A, cont_B, q):
    n_sigmas = 5
    n_points = 200

    A_range = (cont_A['Mean'] - n_sigmas * cont_A['Sigma'],
               cont_A['Mean'] + n_sigmas * cont_A['Sigma'])
    B_range = (cont_B['Mean'] - n_sigmas * cont_B['Sigma'],
               cont_B['Mean'] + n_sigmas * cont_B['Sigma'])
    start = min(A_range[0], B_range[0])
    stop = max(A_range[1], B_range[1])

    xs = np.linspace(start, stop, n_points)
    A_vals = cont_A['Prior'] * norm.pdf(xs, cont_A['Mean'], cont_A['Sigma'])
    B_vals = cont_B['Prior'] * norm.pdf(xs, cont_B['Mean'], cont_B['Sigma'])

    colors = ['r', 'b']
    plt.plot(xs, A_vals, c=colors[0], label='A')
    plt.plot(xs, B_vals, c=colors[1], label='B')

    plt.axvline(x=q['t1'], c='k', lw=0.5, ls=':')
    plt.axvline(x=q['t2'], c='k', lw=0.5, ls=':')

    offset = 0.000007
    sub_level = -0.000025
    left = xs[0]
    right = xs[-1]

    def clip(x, lb, ub):
        res = x
        if res < lb:
            res = lb
        if res > ub:
            res = ub
        return res
    t1 = clip(q['t1'], xs[0], xs[-1])
    t2 = clip(q['t2'], xs[0], xs[-1])

    patch = patches.Rectangle([left, sub_level], t1-left, -sub_level-offset, angle=0.0,
                              color=colors[q['decision'][0]], alpha=0.75)
    plt.gca().add_patch(patch)
    patch = patches.Rectangle([t1, sub_level], t2-t1, -sub_level-offset, angle=0.0,
                              color=colors[q['decision'][1]], alpha=0.75)
    plt.gca().add_patch(patch)
    patch = patches.Rectangle([t2, sub_level], right-t2, -sub_level-offset, angle=0.0,
                              color=colors[q['decision'][2]], alpha=0.75)
    plt.gca().add_patch(patch)
    plt.legend()

    plt.title("Posterior probabilities and strategy q")
    plt.xlabel("image LR feature")
    plt.ylabel("posterior probabilities")


def compute_measurement_lr_discrete(imgs):
    """
    x = compute_measurement_lr_discrete(imgs)

    Calculates difference between left and right half of image(s).

    :param imgs:    set of images, (h, w, n) (or for color images (h, w, 3, n)) np.array
    :return:        (n, ) np.array of values in range <-10, 10>,
    """
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[:, 0:int(width / 2), :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[:, int(width / 2):, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x


def montage(images, colormap='gray'):
    """
    Show images in grid.

    :param images:  np.array (h, w, n)
    """
    h, w, count = np.shape(images)
    h_sq = np.int(np.ceil(np.sqrt(count)))
    w_sq = h_sq
    im_matrix = np.zeros((h_sq * h, w_sq * w))

    image_id = 0
    for k in range(w_sq):
        for j in range(h_sq):
            if image_id >= count:
                break
            slice_w = j * h
            slice_h = k * w
            im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
            image_id += 1
    plt.imshow(im_matrix, cmap=colormap)
    plt.axis('off')
    return im_matrix
