#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import optimize
from numpy import linalg

def perceptron(X, y, max_iterations):
    """
    w, b = perceptron(X, y, max_iterations)

    Perceptron algorithm.
    Implements the perceptron algorithm
    (http://en.wikipedia.org/wiki/Perceptron)

    :param X:               d-dimensional observations, (d, number_of_observations) np array
    :param y:               labels of the observations (0 or 1), (n,) np array
    :param max_iterations:  number of algorithm iterations (scalar)
    :return w:              w - weights, (d,) np array
    :return b:              b - bias, python float
    """
    d, number_of_observations = np.shape(X)
    v = np.zeros(d + 1)  # transformed vector of weights
    Z = np.vstack((np.where(y == 0, X, -X), np.where(y == 0, 1, -1)))  # transformed training set
    # v = Z[:, 0]
    for _ in range(max_iterations):
        # terminal condition
        prediction = np.where(np.dot(v, Z) > 0, True, False)

        # if all inequalities are satisfied -> terminate the algorithm
        if np.all(prediction):
            w, b = v[0:d], v[d]
            return w, b

        # else update vector of weights and continue
        v += Z[:, np.argmax(np.logical_not(prediction))]

    # solution is not found in the given number of iterations
    w, b = float('nan'), float('nan')
    return w, b


def kozinec(X, y, max_iterations):
    """
    w, b = kozinec(X, y, max_iterations)

    Kozinec's algorithm.

    :param X:               d-dimensional observations, (d, number_of_observations) np array
    :param y:               labels of the observations (0 or 1), (n,) np array
    :param max_iterations:  number of algorithm iterations (scalar)
    :return w:              w - weights, (d,) np array
    :return b:              b - bias, python float
    """
    Z = np.vstack((np.where(y == 0, X, -X), np.where(y == 0, 1, -1)))  # transformed training set
    v = Z[:, 0]  # init vector of weights

    for _ in range(max_iterations):
        # terminal condition
        prediction = np.where(np.dot(v, Z) > 0, True, False)

        # if all inequalities are satisfied -> terminate the algorithm
        if np.all(prediction):
            w, b = v[:-1], v[-1]
            return w, b

        # else update vector of weights and continue
        z = Z[:, np.argmax(np.logical_not(prediction))]
        k = np.dot(v, (v - z)[:, None]) / linalg.norm(v - z)**2
        v = (1 - k) * v + k * z

    # solution is not found in the given number of iterations
    w, b = float('nan'), float('nan')
    return w, b


# k2 = optimize.fminbound(lambda kk: linalg.norm((1 - kk) * v + kk * z), -100, 100)
# assert False, '{} {}'.format(k, k2)

def lift_dimension(X):
    """
    Z = lift_dimension(X)

    Lifts the dimensionality of the feature space from 2 to 5 dimensions

    :param X:   observations in the original space
                2-dimensional observations, (2, number_of_observations) np array
    :return Z:  observations in the lifted feature space, (5, number_of_observations) np array
    """

    Z = np.array([X[0], X[1], X[0]*X[0], X[0]*X[1], X[1]*X[1]])
    return Z


def classif_quadrat_perc(tst, model):
    """
    K = classif_quadrat_perc(tst, model)

    Classifies test samples using the quadratic discriminative function

    :param tst:     2-dimensional observations, (2, n) np array
    :param model:   dictionary with the trained perceptron classifier (parameters of the discriminative function)
                        model['w'] - weights vector, np array (d, )
                        model['b'] - bias term, python float
    :return:        Y - classification result (contains either 0 or 1), (n,) np array
    """

    tst = lift_dimension(tst)
    Y = np.where(np.dot(model['w'], tst) + model['b'] > 0, 0, 1)
    return Y


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def pboundary(X, y, model, figsize=None, style_0='bx', style_1='r+'):
    """
    pboundary(X, y, model)

    Plot boundaries for perceptron decision strategy

    :param X:       d-dimensional observations, (d, number_of_observations) np array
    :param y:       labels of the observations (0 or 1), (n,) np array
    :param model:   dictionary with the trained perceptron classifier (parameters of the discriminative function)
                        model['w'] - weights vector, np array (d, )
                        model['b'] - bias term, python float
    """

    plt.figure(figsize=figsize)
    plt.plot(X[0, y == 0], X[1, y == 0], style_0, ms=10)
    plt.plot(X[0, y == 1], X[1, y == 1], style_1, ms=10)

    minx, maxx = plt.xlim()
    miny, maxy = plt.ylim()

    epsilon = 0.1 * np.maximum(np.abs(maxx - minx), np.abs(maxy - miny))

    x_space = np.linspace(minx - epsilon, maxx + epsilon, 1000)
    y_space = np.linspace(miny - epsilon, maxy + epsilon, 1000)
    x_grid, y_grid = np.meshgrid(x_space, y_space)

    x_grid_fl = x_grid.reshape([1, -1])
    y_grid_fl = y_grid.reshape([1, -1])

    X_grid = np.concatenate([x_grid_fl, y_grid_fl], axis=0)
    Y_grid = classif_quadrat_perc(X_grid, model)
    Y_grid = Y_grid.reshape([1000, 1000])

    blurred_Y_grid = ndimage.gaussian_filter(Y_grid, sigma=0)

    plt.contour(x_grid, y_grid, blurred_Y_grid, colors=['black'])
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)

