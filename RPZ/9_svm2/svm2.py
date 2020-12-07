import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg


def get_kernel(Xi, Xj, options):
    """
    Returns kernel matrix K(Xi, Xj)

    :param Xi:      array with features in columns, (d, m) np array
    :param Xj:      array with features in columns, (d, n) np array
    :param options: dict with options
        - options['kernel'] - one of 'rbf', 'linear', 'polynomial'
        - options['sigma']  - sigma for rbf kernel (no need to specify if kernel not rbf), scalar
        - options['d']      - polynom degree for polynomial kernel (no need to specify if kernel not polynomial), scalar
    :return K:      array with kernel function values,  (m, n) np array
    """

    if options['kernel'] == 'linear':
        K = np.dot(Xi.T, Xj)
    elif options['kernel'] == 'polynomial':
        K = (1 + np.dot(Xi.T, Xj)) ** options['d']
    elif options['kernel'] == 'rbf':
        # ||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 * xi^T * xj
        Xi_norm = linalg.norm(Xi, axis=0) ** 2
        Xj_norm = linalg.norm(Xj, axis=0) ** 2
        diff_norm = Xi_norm[:, None] + Xj_norm[None, :] - 2 * np.dot(Xi.T, Xj)
        K = np.exp(- diff_norm / (2 * options['sigma'] ** 2))
    else:
        K = None
    return K


def my_kernel_svm(X, y, C, options):
    """
    Solves kernel soft-margin SVM dual task and outputs model

    :param X:       matrix containing feature points in columns, np array (d, n)
    :param y:       vector with labels (-1, 1) for feature points in X, np array (n, )
    :param C:       regularization constant C, scalar
    :param options: dict that holds options for gsmo QP solver and get_kernel.
        - options['verb']   - verbosity of QP solver, boolean
        - options['t_max']  - maximal number of iterations, scalar [default: np.inf]

        - options['kernel'] - one of 'rbf', 'linear', 'polynomial'
        - options['sigma']  - sigma for rbf kernel (no need to specify if kernel not rbf), scalar
        - options['d']      - polynom degree for polynomial kernel (no need to specify if kernel not polynomial), scalar

    :return model:  dict with entries:
        - model['sv']       - support vectors, np array (d, n_sv)
        - model['y']        - support vectors' labels, np array (n_sv, )
        - model['alpha']    - support vectors' lagrange multipliers, np array (n_sv, )
        - model['options']  - kernel options (same as input)
        - model['b']        - bias term, scalar
        - model['fun']      - function that should be used for classification, function reference
    """

    n = y.size
    H = get_kernel(X, X, options) * np.dot(y[:, None], y[None, :])
    f = - np.ones(n)

    options_q = {key: options[key] for key in ['verb', 't_max']}
    alphas, _, _, finished = gsmo(H=H, f=f, a=y, b=0, lb=np.zeros(n), ub=C * np.ones(n), x_init=np.zeros(n),
                                  **options_q)
    if not finished:
        return None

    sv_idx = np.arange(n)[(alphas >= 1e-10) & (alphas <= C)]

    model = dict()
    model['sv'] = X[:, sv_idx]
    model['y'] = y[sv_idx]
    model['alpha'] = alphas[(alphas >= 1e-10) & (alphas <= C)]
    model['options'] = options
    model['b'] = compute_bias(get_kernel(X, X, options), y, alphas, C)
    model['fun'] = classif_kernel_svm
    return model


def classif_kernel_svm(X, model):
    """
    Performs classification on X by trained SVM classifier stored in model

    :param X:        matrix containing feature points in columns, np array (d, n)
    :param model:    dict with SVM model (my_kernel_svm output)

    :return classif: labels (-1, 1) for feature points in X, np array (n, )
    """

    classif = np.sign(np.dot(model['alpha'] * model['y'], get_kernel(model['sv'], X, model['options'])) + model['b'])
    return classif


def compute_kernel_test_error(itrn, itst, X, y, C, options):
    """
    Computes mean risk computed over crossvalidation folds (train svm on train set and evaluate on test set)

    :param itrn:    list with training data indices folds (from crossval function)
    :param itst:    list with testing data indices folds (from crossval function)
    :param X:       input feature vector, (n, m) np array
    :param y:       labels 1, -1, (m, ) np array
    :param C:       regularization constant C, scalar
    :param options: dict with options for my_kernel_svm and classif_kernel_svm

    :return error:  mean test error
    """

    num_fold = len(itrn)
    error = np.zeros(num_fold)
    for i in range(num_fold):
        model = my_kernel_svm(X[:, itrn[i]], y[itrn[i]], C, options)
        classified_labels = classif_kernel_svm(X[:, itst[i]], model)
        error[i] = np.mean(classified_labels != y[itst[i]])
    return np.mean(error)


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def gsmo(H, f, a, b=0, lb=-np.inf, ub=np.inf, x_init=None, nabla_0=None, tolerance_KKT=0.001, verb=0, t_max=np.inf,
         options=None):
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

    :param options:
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


def compute_bias(K, ys, alphas, C):
    """
    Computes SVM bias from dual solution.

    See https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf (Section 4.1.5)
    See http://fouryears.eu/wp-content/uploads/svm_solutions.pdf (Exercise 4) - "easy to follow" explanation

    :param K:       matrix with kernel function values,  (n, n) np array
    :param ys:      vector with labels (-1, 1) for feature points in X, np array (n, )
    :param alphas:  vector with SVM dual problem solution alphas, np array (n, )
    :param C:       regularization constant C, scalar

    :return b:      the bias term, scalar
    """
    N = ys.size
    eps = 1e-10
    alpha_y = np.reshape(alphas * ys, (len(alphas), 1))

    # check if there are support vectors exactly on the margin
    on_margin_mask = np.logical_and(alphas > eps, alphas < C)
    if np.any(on_margin_mask):
        # there are some! lets do the obvious bias computation
        ids = np.nonzero(on_margin_mask)[0]

        bias = np.mean(ys[ids] - np.sum(alpha_y * K[:, ids],
                                        axis=0))

        ## Vectorized version of:
        # slow_bias = 0
        # for svi in ids:
        #     s = 0
        #     for i in range(N):
        #         s += alphas[i] * ys[i] * K[i, svi]

        #     slow_bias += ys[svi] - s
        # slow_bias /= len(ids)
        # assert np.isclose(bias, slow_bias)
    else:
        # no support vectors exactly on the margin
        sv_mask = alphas > eps
        not_sv_mask = np.logical_not(sv_mask)

        sv_indices = np.nonzero(sv_mask)[0]
        e_i = ys - np.sum(alpha_y * K, axis=0)

        ## Vectorized version of:
        # def e(i):
        #     wx_i = 0
        #     for svi in sv_indices:
        #         wx_i += alphas[svi] * ys[svi] * K[i, svi]
        #     return ys[i] - wx_i
        # slow_e_i = np.array([e(i) for i in range(len(alphas))])
        # assert np.allclose(e_i, slow_e_i)

        LB_mask = np.logical_or(
            np.logical_and(not_sv_mask, ys == 1),
            np.logical_and(sv_mask, ys == -1))
        LB = np.amax(e_i[LB_mask])

        UB_mask = np.logical_or(
            np.logical_and(sv_mask, ys == 1),
            np.logical_and(not_sv_mask, ys == -1))
        UB = np.amin(e_i[UB_mask])

        # anything between LB and UB would be correct
        # lets just pick their mean
        bias = (LB + UB) / 2

    return bias


def montage(images, colormap='gray'):
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
        imgs = test_images[:, :, labels == i]
        subfig = plt.subplot(1, len(letters), i + 1)
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


def plot_boundary(ax, model):
    """
    Plots 2-class linear decision boundary

    :param ax:      axes to draw onto, matplotlib Axes
    :param model:   dictionary returned by my_kernel_svm
    """

    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()

    xs = np.linspace(x_lim[0], x_lim[1], 400)
    ys = np.linspace(y_lim[0], y_lim[1], 400)

    xs, ys = np.meshgrid(xs, ys, indexing='ij')
    X = np.vstack((xs.flatten(), ys.flatten()))

    z = model['fun'](X, model).reshape(xs.shape)
    plt.contour(xs, ys, z, [0])


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
