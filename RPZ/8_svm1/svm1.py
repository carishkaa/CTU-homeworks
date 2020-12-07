#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def my_svm(X, y, C, options=None):
    """
    Solves linear soft-margin SVM dual task and outputs w, b and support vectors indices

    :param X:       matrix containing feature points in columns, np array (d, n)
    :param y:       vector with labels (-1, 1) for feature points in X, np array (n, )
    :param C:       regularization constant C, scalar
    :param options: (optional) dict that holds options for gsmo QP solver fields:
                    options['verb'] (verbosity of QP solver),
                    options['tmax'] (maximal number of iterations, default is inf)
    :return w:      normal vector of the separating hyperplane, np array (d, )
    :return b:      the bias term, scalar
    :return sv_idx: vector with indices of support vectors, np array (p, )
    """

    n = y.size
    if options is None:
        options = {}
    options['verb'] = options['verb'] if 'verb' in options else True
    options['tmax'] = options['tmax'] if 'tmax' in options else np.inf

    H = np.dot(X.T, X) * np.dot(y.reshape(-1, 1), y.reshape(1, -1))
    f = - np.ones(n)

    alphas, fval, t, finished = gsmo(H=H, f=f, a=y, b=0, lb=np.zeros(n), ub=C*np.ones(n), x_init=np.zeros(n))
    w = np.sum(X*alphas*y, axis=1)
    b = compute_bias(X, y, alphas, C)
    sv_idx = np.arange(n)[alphas > 1e-10]
    return w, b, sv_idx


def classif_linear_svm(X, model):
    """
    classifies data using the provided linear model

    :param X:        matrix containing feature points in columns, np array (d, n)
    :param model:    model represented as dict with keys 'w' and 'b', dict
        - model['w'] - model weights, np array (d, )
        - model['b'] - representing model bias, scalar

    :return classif: labels (-1, 1) for feature points in X, np array (n, )

    """

    classif = np.where(np.dot(X.T, model['w']) + model['b'] > 0, 1, -1)
    return classif


def compute_measurements_2d(data, normalization=None):
    """
    Computes 2D features from image measurements

    :param data:          dict with keys 'images' and 'labels'
        - data['images'] - (h, w, n) np.uint8 array of images
        - data['labels'] - (n, ) np array of image labels (-1 or 1)
    :param normalization: - optional - dict with keys 'a_mean', 'a_std', 'b_mean', 'b_std'
        - normalization['a_mean'] - mean of the LR feature
        - normalization['a_std'] - std of the LR feature
        - normalization['b_mean'] - mean of the UD feature
        - normalization['b_std'] - std of the UD feature

    :return:              X - (2, n) np array of features
                          y - (n, ) np array of image labels (1 and -1)
                          normalization - same as :param normalization:,
                                          computed from data if normalization parameter not provided
    """
    imgs = data['images']
    y = data['labels']
    lr = compute_measurement_lr_cont(imgs)
    ul = compute_measurement_ul_cont(imgs)

    if not normalization:
        normalization = dict()
        normalization['a_mean'], normalization['a_std'] = lr.mean(), lr.std()
        normalization['b_mean'], normalization['b_std'] = ul.mean(), ul.std()

    X = np.array([(lr - normalization['a_mean']) / normalization['a_std'],
                  (ul - normalization['b_mean']) / normalization['b_std']])

    return X, y, normalization


def compute_test_error(itrn, itst, X, y, C):
    """
    Computes mean risk computed over crossvalidation folds (train svm on train set and evaluate on test set)

    :param itrn:  list with training data indices folds (from crossval function)
    :param itst:  list with testing data indices folds (from crossval function)
    :param X:     input feature vector, (n, m) np array
    :param y:     labels 1, -1, (m, ) np array
    :param C:     regularization constant C, scalar

    :return mean: mean test error
    """
    n_fold = len(itrn)
    error = np.zeros(n_fold)
    for i in range(n_fold):
        w, b, _ = my_svm(X[:, itrn[i]], y[itrn[i]], C)
        classified_labels = classif_linear_svm(X[:, itst[i]], {'w': w, 'b': b})
        error[i] = np.mean(classified_labels != y[itst[i]])

    return np.mean(error)


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def gsmo(H, f, a, b=0, lb=-np.inf, ub=np.inf, x_init=None, nabla_0=None, tolerance_KKT=0.001, verb=0, t_max=np.inf):
    """
    GSMO Generalized SMO algorithm for classifier design.

    Reimplemented to Python from:
    https://gitlab.fel.cvut.cz/smidm/statistical-pattern-recognition-toolbox

    Description:
        This function implements the generalized SMO algorithm which solves
        the following QP task:

        min_x Q_P(x) = 0.5*x'*H*x + f'*x

        s.t.  a'*x = b
              lb(i) <= x(i) <= ub(i)   for all i=1:n

    Reference:
        S.S.Keerthi, E.G.Gilbert: Convergence of a generalized SMO algorithm for SVM
        classifier design. Machine Learning, Vol. 46, 2002, pp. 351-360.

    :param H:               symmetric positive semidefinite matrix, np array (n, n)
    :param f:               vector, np array (n, )
    :param a:               vector which must not contain zero entries, np array (n, )
    :param b:               scalar [default: 0]
    :param lb:              lower bound, np array (n, ) [default: -inf]
    :param ub:              upper bound, np array (n, ) [default: +inf]
    :param x_init:          initial solution, np array (n, )
    :param nabla_0:         Nabla0 = H*x0 + f, np array (n, )
    :param tolerance_KKT:   Determines relaxed KKT conditions. It correspondes to $\tau$ in Keerthi's paper.
                            scalar [default: tolKKT=0.001]
    :param verb:            if > 0 then prints info every verb-th iterations, scalar [default: 0]
    :param t_max:           maximal number of iterations, scalar [default: +inf]

    :return x:              solution vector, np array (n, )
    :return fval:           atteined value of the optimized QP criterion fval=Q_P(x), scalar
    :return t:              number of iterations, scalar
    :return finished:       True if an optimal solution was found, boolean
    """

    # Setup
    if (np.isscalar(lb)):
        lb = np.array([lb] * len(f))

    if (np.isscalar(ub)):
        ub = np.array([ub] * len(f))

    if (x_init is None):
        # Find feasible x0
        x_init = np.zeros(len(f))
        xa = 0
        i = 0
        while (not np.allclose(np.dot(x_init, a), b)):
            if (i >= len(a)):
                raise ValueError("Constraints cannot be satisfied")
            x_init[i] = np.clip((b - xa) / a[i], lb[i], ub[i])
            xa += x_init[i] * a[i]
            i += 1
    else:
        x_init = np.clip(x_init, lb, ub)

    if (nabla_0 is None):
        nabla_0 = np.dot(H, x_init) + f

    # Initialization
    t = 0
    finished = False
    x = np.copy(x_init)
    nabla = np.copy(nabla_0)

    # SMO
    while (t < t_max):
        assert np.allclose(x, np.clip(x, lb, ub))  # x0 within bounds

        # Find the most violating pair of variables
        (minF, minI) = (np.inf, 0)
        (maxF, maxI) = (-np.inf, 0)
        F = nabla / a
        x_feas = np.logical_and(lb < x, x < ub)
        a_pos = a > 0
        a_neg = a < 0

        x_lb = x == lb
        x_ub = x == ub

        min_mask = np.logical_or.reduce((x_feas, np.logical_and(x_lb, a_pos), np.logical_and(x_ub, a_neg)))
        max_mask = np.logical_or.reduce((x_feas, np.logical_and(x_lb, a_neg), np.logical_and(x_ub, a_pos)))

        if np.sum(max_mask) > 0:
            tmp = F.copy()
            tmp[np.logical_not(max_mask)] = -np.inf
            maxI = np.argmax(tmp)
            maxF = tmp[maxI]

        if np.sum(min_mask) > 0:
            tmp = F.copy()
            tmp[np.logical_not(min_mask)] = np.inf
            minI = np.argmin(tmp)
            minF = tmp[minI]

        # Check KKT conditions
        if (maxF - minF <= tolerance_KKT):
            finished = True
            break

        # SMO update the most violating pair
        tau_lb_u = (lb[minI] - x[minI]) * a[minI]
        tau_ub_u = (ub[minI] - x[minI]) * a[minI]
        if (a[minI] <= 0):
            (tau_lb_u, tau_ub_u) = (tau_ub_u, tau_lb_u)

        tau_lb_v = (x[maxI] - lb[maxI]) * a[maxI]
        tau_ub_v = (x[maxI] - ub[maxI]) * a[maxI]
        if (a[maxI] > 0):
            (tau_lb_v, tau_ub_v) = (tau_ub_v, tau_lb_v)

        tau = (nabla[maxI] / a[maxI] - nabla[minI] / a[minI]) / \
              (H[minI, minI] / a[minI] ** 2 + H[maxI, maxI] / a[maxI] ** 2 - 2 * H[maxI, minI] / (a[minI] * a[maxI]))

        tau = min(max(tau, tau_lb_u, tau_lb_v), tau_ub_u, tau_ub_v)
        if (tau == 0):
            # Converged on a non-optimal solution
            break

        x[minI] += tau / a[minI]
        x[maxI] -= tau / a[maxI]

        nabla += H[:, minI] * tau / a[minI]
        nabla -= H[:, maxI] * tau / a[maxI]

        t += 1
        # Print iter info
        if (verb > 0 and t % verb == 0):
            obj = 0.5 * np.sum(x * nabla + x * f)
            print("t=%d, KKTviol=%f, tau=%f, tau_lb=%f, tau_ub=%f, Q_P=%f" % \
                  (t, maxF - minF, tau, max(tau_lb_u, tau_lb_v), min(tau_ub_u, tau_ub_v), obj))
            # raw_input()

    fval = 0.5 * np.dot(x, nabla + f)
    return x, fval, t, finished


def compute_bias(xs, ys, alphas, C):
    """
    Computes SVM bias from dual solution.

    See https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf (Section 4.1.5)
    See http://fouryears.eu/wp-content/uploads/svm_solutions.pdf (Exercise 4) - "easy to follow" explanation

    :param xs:      matrix containing feature points in columns, np array (d, n)
    :param ys:      vector with labels (-1, 1) for feature points in X, np array (n, )
    :param alphas:  vector with SVM dual problem solution alphas, np array (n, )
    :param C:       regularization constant C, scalar

    :return b:      the bias term, scalar
    """
    N = ys.size
    eps = 1e-10

    # check if there are support vectors exactly on the margin
    on_margin_mask = np.logical_and(alphas > eps, alphas < C)
    if np.any(on_margin_mask):
        # there are some! lets do the obvious bias computation
        ids = np.nonzero(on_margin_mask)[0]

        bias = 0
        for svi in ids:
            s = 0
            for i in range(N):
                s += alphas[i] * ys[i] * np.dot(xs[:, i], xs[:, svi])

            bias += ys[svi] - s
        bias /= len(ids)
    else:
        # no support vectors exactly on the margin
        sv_mask = alphas > eps
        not_sv_mask = np.logical_not(sv_mask)

        sv_indices = np.nonzero(sv_mask)[0]

        def e(i):
            wx_i = 0
            for svi in sv_indices:
                wx_i += alphas[svi] * ys[svi] * np.dot(xs[:, svi], xs[:, i])
            return ys[i] - wx_i

        LB_mask = np.logical_or(
            np.logical_and(not_sv_mask, ys == 1),
            np.logical_and(sv_mask, ys == -1))
        LB_ids = np.nonzero(LB_mask)[0]
        LB = np.amax([e(i) for i in LB_ids])

        UB_mask = np.logical_or(
            np.logical_and(sv_mask, ys == 1),
            np.logical_and(not_sv_mask, ys == -1))
        UB_ids = np.nonzero(UB_mask)[0]
        UB = np.amin([e(i) for i in UB_ids])

        # anything between LB and UB would be correct
        # lets just pick their mean
        bias = (LB + UB) / 2

    return bias


def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np array (h, w, n)
    :param labels:          labels for input images np array (n,)
    :param letters:         string with letters, e.g. 'CN'
    """

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = np.shape(images)
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

    for i in range(len(letters)):
        imgs = test_images[:,:,labels==i]
        subfig = plt.subplot(1,len(letters),i+1)
        montage(imgs)
        plt.title(letters[i])


def crossval(num_data, num_folds):
    """
    itrn, itst = crossval(num_data, num_folds)

    Partitions data for cross-validation.

    This function randomly partitions data into the training
    and testing parts. The number of partitioning is determined
    by the num_folds. If num_folds==1 then makes only one random
    partitioning of data into training and testing in ratio 50:50.

    :param num_data:    number of data (scalar, integer)
    :param num_folds:   number of folders (scalar, integer)

    :return itrn:       LIST of training folds
                         itrn[i] - (n, ) np.array of i-th fold indices into the training data
    :return itst:       LIST of testing folds
                         itst[i] - (n, ) np.array of i-th fold indices into the testing data
    """
    if num_folds < 2:
        print("Minimal number of folds set to 2.")
        num_folds = 2

    inx = np.random.permutation(num_data)

    itrn = []
    itst = []

    num_column = np.int32(np.ceil(np.float64(num_data) / num_folds))

    for idx in range(num_folds):
        tst_idx = range((idx * num_column), np.min([num_data, ((idx + 1) * num_column)]))
        trn_idx = [i for i in list(range(num_data)) if i not in tst_idx]
        itst.append(inx[tst_idx])
        itrn.append(inx[trn_idx])
    return itrn, itst


def plot_points(X, y, size=None):
    """
    Plots 2D points from two classes

    :param X: input data, np array (2, n)
    :param y: class labels (classes -1, +1), np array (n, )
    """
    y = np.squeeze(y)
    pts_A = X[:, y > 0]
    pts_B = X[:, y < 0]

    plt.scatter(pts_A[0, :], pts_A[1, :], s=size)
    plt.scatter(pts_B[0, :], pts_B[1, :], s=size)


def plot_boundary(ax, w, b, support_vectors=None):
    """
    Plots 2-class linear decision boundary

    :param ax:              axes to draw onto, matplotlib Axes
    :param w:               model weights, np array (2, )
    :param b:               model bias, scalar
    :param support_vectors: (optional) coordinates of the support vectors, np array (2, n)
    :return:
    """

    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()

    xs = np.linspace(x_lim[0], x_lim[1])
    ys = np.linspace(y_lim[0], y_lim[1])

    xs, ys = np.meshgrid(xs, ys, indexing='ij')

    z = xs * w[0] + ys * w[1] + b
    plt.contour(xs, ys, z, [-1, 0, 1], colors=['k', 'r', 'k'], linestyles=['dashed', 'solid', 'dashed'])

    if support_vectors is not None:
        sv_scale = (plt.rcParams['lines.markersize'] ** 2) * 5
        plt.scatter(support_vectors[0, :],
                    support_vectors[1, :],
                    s=sv_scale,
                    facecolors='none',
                    edgecolors='k',
                    marker='o')


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

    Compute measurement on images, subtract sum of top half from sum of
    bottom half.

    :param imgs:    set of images, (h, w, n) numpy array
    :return x:      measurements, (n, ) numpy array
    """
    assert len(imgs.shape) == 3

    h = imgs.shape[0]
    sum_columns = np.sum(imgs, dtype=np.float64, axis=1)

    x = np.sum(sum_columns[0:int(h / 2), :], axis=0) - np.sum(sum_columns[int(h / 2):, :], axis=0)

    assert x.shape == (imgs.shape[2],)
    return x
