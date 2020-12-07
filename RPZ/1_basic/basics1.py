#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def matrix_manip(A, B):
    """
    output = matrix_manip(A,B)

    Perform example matrix manipulations.

    :param A: np.array (k, l), l >= 3
    :param B: np.array (2, n)
    :return:
       output['A_transpose'] (l, k), same dtype as A
       output['A_3rd_col'] (k, 1), same dtype as A
       output['A_slice'] (2, 3), same dtype as A
       output['A_gr_inc'] (k, l+1), same dtype as A
       output['C'] (k, k), same dtype as A
       output['A_weighted_col_sum'] python float
       output['D'] (2, n), same dtype as B
       output['D_select'] (2, n'), same dtype as B
    """
    output = dict()

    # 1) Find the transpose of the matrix A and return it in output['A_transpose']
    output['A_transpose'] = np.transpose(A)

    # 2) Select the third column of the matrix A and return it in output['A_3rd_col'].
    output['A_3rd_col'] = np.transpose([A[:, 2]])

    # 3) Select last two rows from last three columns of the matrix A
    output['A_slice'] = A[-2:, -3:]

    # 4) Find all positions in A greater then 3 and increment them by 1.
    # Afterwards add a new column of ones to the matrix (from right).
#    A2 = np.array(A, copy=True)
#    A2[A2 > 3] += 1
#    mat_ones = np.ones((m, 1))
#    output['A_gr_inc'] = np.append(A2, mat_ones, axis=1)
    n = A.shape[0]
    m = A.shape[1]
    output['A_gr_inc'] = np.ones((n, m + 1), dtype=A.dtype)
    output['A_gr_inc'][:, :-1] = A + (A > 3)

    # 5)
    output['C'] = np.dot(output['A_gr_inc'], np.transpose(output['A_gr_inc']))

    # 6)
    m = output['A_gr_inc'].shape[1]  # pocet sloupcu
    output['A_weighted_col_sum'] = float(np.dot(np.arange(1, m+1), output['A_gr_inc'].sum(axis=0)))

    # 7) Subtract a vector (4 , 6) from all columns of matrix B
    output['D'] = B - np.transpose([[4, 6]])

    # 8) Select all column vectors in the matrix D, which have greater euclidean length
    # than the average length of column vectors in D
    euclidean_lengths = np.sqrt(np.square(output['D']).sum(axis=0))
    average = np.average(euclidean_lengths)
    output['D_select'] = output['D'][:, euclidean_lengths > average]

    return output


def compute_letter_mean(letter_char, alphabet, images, labels):
    """
    img = compute_letter_mean(letter_char, alphabet, images, labels)

    Compute mean image for a letter.

    :param letter_char:  character, e.g. 'm'
    :param alphabet:     np.array of all characters present in images (n_letters, )
    :param images:       images of letters, np.array of size (H, W, n_images)
    :param labels:       image labels, np.array of size (n_images, ) (index into alphabet array)
    :return:             mean of all images of the letter_char, (H, W) np.uint8 dtype (round, then convert)
    """
    letter_images = images[:, :, labels == np.where(alphabet == letter_char)[0]]
    letter_mean = np.uint8(np.round(np.average(letter_images, axis=2)))
    return letter_mean


def compute_lr_histogram(letter_char, alphabet, images, labels, num_bins, return_bin_edges=False):
    """
    lr_histogram = compute_lr_histogram(letter_char, alphabet, images, labels, num_bins)

    Calculates feature histogram.

    :param letter_char:                is a character representing the letter whose feature histogram
                                       we want to compute, e.g. 'C'
    :param alphabet:                   np.array of all characters present in images (n_letters, )
    :param images:                     images of letters, np.array of shape (h, w, n_images)
    :param labels:                     image_labels, np.array of size (n_images, ) (indexes into alphabet array)
    :param num_bins:                   number of histogram bins
    :param return_bin_edges:
    :return:                           counts of values in the corresponding bins, np.array (num_bins, ),
                                       (only if return_bin_edges is True) histogram bin edges, np.array (num_bins+1, )
    """
    # letter_images = images[:, :, labels == np.where(alphabet == letter_char)[0]]
    # w = letter_images.shape[1]
    #
    # left_halves = letter_images[:, 0:w//2, :]
    # right_halves = letter_images[:, w//2:, :]
    #
    # left_sum = np.sum(np.sum(left_halves, axis=1), axis=0)
    # right_sum = np.sum(np.sum(right_halves, axis=1), axis=0)
    #
    # x_values = np.subtract(left_sum, right_sum, dtype=int)
    # lr_histogram, bin_edges = np.histogram(x_values, bins=num_bins)

    if return_bin_edges:
        return lr_histogram, bin_edges
    else:
        return lr_histogram


def histogram_plot(hist_data, color, alpha):
    """
    Plot histogram from outputs of compute_lr_histogram

    :param hist_data: tuple of (histogram values, histogram bin edges)
    :param color:     color of the histogram (passed to matplotlib)
    :param alpha:     transparency alpha of the histogram (passed to matplotlib)
    """
    hist, bin_edges = hist_data
    axis = range(len(hist))
    return plt.bar(axis, hist, alpha=alpha, color=color)

################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def montage(images, colormap='gray'):
    """
    Show images in grid.

    :param images:  np.array (h x w x n_images)
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
