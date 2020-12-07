#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def logistic_loss(X, y, w):
    """
    E = logistic_loss(X, y, w)

    Evaluates the logistic loss function.

    :param X:    d-dimensional observations with X[0, :] == 1, np.array (d, n)
    :param y:    labels of the observations, np.array (n, )
    :param w:    weights, np.array (d, )
    :return E:   calculated loss, python float
    """
    kwx = y * np.dot(w, X)
    E = float(np.mean(np.log(1 + np.exp(-kwx))))
    return E


def logistic_loss_gradient(X, y, w):
    """
    g = logistic_loss_gradient(X, y, w)

    Calculates gradient of the logistic loss function.

    :param X:   d-dimensional observations with X[0, :] == 1, np.array (d, n)
    :param y:   labels of the observations, np.array (n, )
    :param w:   weights, np.array (d, )
    :return g:  resulting gradient vector, np.array (d, )
    """
    kwx = y * np.dot(w, X)
    g_kwx = -np.multiply(X, y)
    g = np.mean(g_kwx/(1 + np.exp(kwx)), axis=1)
    return g


def logistic_loss_gradient_descent(X, y, w_init, epsilon):
    """
    w, wt, Et = logistic_loss_gradient_descent(X, y, w_init, epsilon)

    Performs gradient descent optimization of the logistic loss function.

    :param X:       d-dimensional observations with X[0, :] == 1, np.array (d, n)
    :param y:       labels of the observations, np.array (n, )
    :param w_init:  initial weights, np.array (d, )
    :param epsilon: parameter of termination condition: np.norm(w_new - w_prev) <= epsilon, python float
    :return w:      w - resulting weights, np.array (d, )
    :return wt:     wt - progress of weights, np.array (d, number_of_accepted_candidates)
    :return Et:     Et - progress of logistic loss, np.array (number_of_accepted_candidates, )
    """
    # init
    step_size = 1.0
    E = logistic_loss(X, y, w_init)
    g = logistic_loss_gradient(X, y, w_init)
    wt = np.array([w_init])
    Et = np.array([E])
    w = w_init
    w_prev = 0

    i_iter = 0
    # iterate
    while np.linalg.norm(w - w_prev, 2) > epsilon:
        i_iter += 1
        E_new = logistic_loss(X, y, w - step_size * g)
        g_new = logistic_loss_gradient(X, y, w - step_size * g)
        if i_iter == 100000:
            return w, wt.T, Et
            # assert False, '{}'.format(w)
        if E_new < E:
            w_prev = w.copy()
            w -= step_size * g
            E = E_new
            g = g_new
            wt = np.vstack((wt, w))
            Et = np.append(Et, E)
            step_size *= 2
        else:
            step_size /= 2
    return w, wt.T, Et


def classify_images(X, w):
    """
    y = classify_images(X, w)

    Classification by logistic regression.

    :param X:    d-dimensional observations with X[0, :] == 1, np.array (d, n)
    :param w:    weights, np.array (d, )
    :return y:   estimated labels of the observations, np.array (n, )
    """
    y = np.dot(w, X)
    y = np.where(y >= 0, 1, -1)
    return y


def get_threshold(w):
    """
    thr = get_threshold(w)

    Returns the optimal decision threshold given the sigmoid parameters w (for 1D data).

    :param w:    weights, np.array (2, )
    :return:     calculated threshold (scalar)
    """
    # wx = 0 -> w1*x + w0 = 0 -> x = -w0/w1
    thr = -w[0]/w[1]
    return thr


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def plot_gradient_descent(X, y, loss_function, w, wt, Et, min_w=-10, max_w=10, n_points=20):
    """
    plot_gradient_descent(X, y, loss_function, w, wt, Et)

    Plots the progress of the gradient descent.

    :param X:               d-dimensional observations with X[0, :] == 1, np.array (d, n)
    :param y:               labels of the observations, np.array (n, )
    :param loss_function:   pointer to a logistic loss function
    :param w:               weights, np.array (d, )
    :param wt:              progress of weights, np.array (d, number_of_accepted_candidates)
    :param Et:              progress of logistic loss, np.array (number_of_accepted_candidates, )
    :return:
    """

    if X.shape[0] != 2:
        raise NotImplementedError('Only 2-d loss functions can be visualized using this method.')

    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca()

    # Plot the gradient descent
    W1, W2 = np.meshgrid(np.linspace(min_w, max_w, n_points), np.linspace(min_w, max_w, n_points))
    L = np.zeros_like(W1)
    for i in range(n_points):
        for j in range(n_points):
            L[i, j] = loss_function(X, y, np.array([W1[i, j], W2[i, j]]))
    z_min, z_max = np.min(L), np.max(L)
    c = ax.pcolor(W1, W2, L, cmap='viridis', vmin=z_min, vmax=z_max, edgecolor='k')
    fig.colorbar(c, ax=ax)

    # Highlight the found minimum
    plt.plot([w[0]], w[1], 'rs', markersize=15, fillstyle='none')
    plt.plot(wt[0, :], wt[1, :], 'w.', markersize=15, linewidth=1)
    plt.plot(wt[0, :], wt[1, :], 'w-', markersize=15, linewidth=1)
    plt.xlim([min_w, max_w])
    plt.ylim([min_w, max_w])
    plt.xlabel('w_0')
    plt.ylabel('w_1')
    plt.title('Gradient descent')


def plot_aposteriori(X, y, w):
    """
    plot_aposteriori(X, y, w)

    :param X:    d-dimensional observations with X[0, :] == 1, np.array (d, n)
    :param y:    labels of the observations, np.array (n, )
    :param w:    weights, np.array (d, )
    """


    xA = X[:,y == 1]
    xC = X[:,y == -1]

    plot_range = np.linspace(np.min(X[1,:]) - 0.5, np.max(X[1,:]) + 0.5, 100)
    pAx = 1 / (1 + np.exp(-plot_range * w[1] - w[0]))
    pCx = 1 / (1 + np.exp(plot_range * w[1] + w[0]))

    thr = get_threshold(w)

    plt.figure()
    plt.plot(plot_range, pAx, 'b-', LineWidth=2)
    plt.plot(plot_range, pCx, 'r-', LineWidth=2)
    plt.plot(xA, np.zeros_like(xA), 'b+')
    plt.plot(xC, np.ones_like(xC), 'r+')
    plt.plot([thr, thr], [0, 1], 'k-')
    plt.legend(['p(A|x)', 'p(C|x)'])


def compute_measurements(imgs, norm_parameters=None):
    """
    x = compute_measurement(imgs [, norm_parameters])

    Compute measurement on images, subtract sum of right half from sum of
    left half.

    :param imgs:              input images, np array (h, w, n)
    :param norm_parameters:   norm_parameters['mean'] python float
                              norm_parameters['std']  python float
    :return x:                measurements, np array (n, )
    :return norm_parameters:  norm_parameters['mean'] python float
                              norm_parameters['std']  python float
    """


    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    left_half  = np.sum(sum_rows[:int(width // 2),:], axis=0)
    right_half = np.sum(sum_rows[int(width // 2):,:], axis=0)
    x = left_half - right_half

    if norm_parameters is None:
        # If normalization parameters are not provided, compute it from data
        norm_parameters = {'mean': float(np.mean(x)), 'std': float(np.std(x))}

    x = (x - norm_parameters['mean']) / norm_parameters['std']

    return x, norm_parameters


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np.array (h, w, n)
    :param labels:          labels for input images np.array (n,)
    :param letters:         string with letters, e.g. 'CN'
    """

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np.array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = images.shape
        h_sq = np.int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.zeros((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
                image_id += 1
        plt.imshow(im_matrix, cmap=colormap)
        plt.axis('off')
        return im_matrix

    unique_labels = np.unique(labels).flatten()
    for i in range(len(letters)):
        imgs = test_images[:,:,labels==unique_labels[i]]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])




def show_mnist_classification(imgs, labels, imgs_shape=None):
    """
    function show_mnist_classification(imgs, labels)

    Shows results of MNIST digits classification.

    :param imgs:         flatten images - d-dimensional observations, np.array (height x width, n)
    :param labels:       labels for input images np.array (n,)
    :param imgs_shape:   image dimensions, np.array([height, width])
    :return:
    """

    if imgs_shape is None:
        imgs_shape = np.array([28, 28])

    n_images = imgs.shape[1]
    images = np.zeros([imgs_shape[0], imgs_shape[1], n_images])

    for i in range(n_images):
        images[:, :, i] = np.reshape(imgs[:, i], [imgs_shape[0], imgs_shape[1]])

    plt.figure(figsize=(20,10))
    show_classification(images, labels, '01')


if __name__ == "__main__":
    jj = 0