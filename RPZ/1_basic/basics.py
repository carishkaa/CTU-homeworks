#!/usr/bin/python
# -*- coding: utf-8 -*-
from gettext import find
from math import sqrt

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

    output1 = dict()
    output1['A_transpose'] = np.transpose(A)
    output1['A_3rd_col'] = A[:, 2]
    # Select last two rows from last three columns of the matrix A
    (m, n) = np.shape(A)
    col_slice = A[:, [n - 3, n - 2, n - 1]]
    output1['A_slice'] = col_slice[[m - 2, m - 1], :]

    # Find all positions in A greater then 3 and increment them by 1. Afterwards add a new column of ones to the matrix (from right)

    A2 = np.array(A, copy=True)
    A2[A2 > 3] += 1
    mat_ones = np.ones((m, 1))
    output['A_gr_inc'] = np.append(A2, mat_ones, axis=1)


    # soucin matic?
    # dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    (k, l) = np.shape(output1['A_gr_inc'])
    output1['C'] = np.dot(output1['A_gr_inc'], np.transpose(output1['A_gr_inc']))

    # np.arange(start, stop, step - optional)
    (r, c) = np.shape(output1['A_gr_inc'])
    # [1...c]
    sum_c = (np.arange(1, c + 1))
    # soucin prvku v sloupcich [...]
    sum_A_gr = np.sum(output1['A_gr_inc'], axis=0)

    final_sum = np.dot(sum_c, np.transpose(sum_A_gr))
    output1['A_weighted_col_sum'] = float(final_sum)

    # Subtract a vector (4,6)t from all columns of matrix B
    vector = np.array([4, 6])
    D = B - np.vstack(vector)
    output1['D'] = D

    # Select all column vectors in the matrix D,
    # which have greater euclidean length than the
    # average length of column vectors in D

    final = np.linalg.norm(D, axis=0)
    average = np.sum(final) / len(D)

    # B = np.delete(B, 2, 0)   delete third row of B
    # C = np.delete(C, 1, 1)   delete second column of C
    # D(:, find(sqrt(sum(D^ 2)) > average)

    for i in range(len(final) - 1, -1, -1):
        if final[i] <= average:
            output1['D_select'] = np.delete(D, i, 1)

    print(output1)
    return output1


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
    #loaded_data = np.load("data_33rpz_basics.npz")

    letter_mean = np.mean(images[:, :, labels == np.where(alphabet == letter_char)[0]], 2)
    rou = np.round(letter_mean)
    final = np.uint8(rou)

    return final


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

    images_om = images[:, :, labels == np.where(alphabet == letter_char)[0]]

    sum_left = np.sum(np.sum(images_om[:, :5, :],axis = 0), axis = 0)
    sum_right = np.sum(np.sum(images_om[:, 5:, :], axis = 0), axis = 0)

    x = np.subtract(sum_left, sum_right, dtype = int)

    lr_histogram, bin = np.histogram(x, bins = num_bins)

    #print(lr_histogram)
    return lr_histogram


def histogram_plot(hist_data, color, alpha):
    """
    Plot histogram from outputs of compute_lr_histogram

    :param hist_data: tuple of (histogram values, histogram bin edges)
    :param color:     color of the histogram (passed to matplotlib)
    :param alpha:     transparency alpha of the histogram (passed to matplotlib)
    """
    loaded_data = np.load("data_33rpz_basics.npz")
    alphabet = loaded_data['alphabet']
    images = loaded_data['images']
    labels = loaded_data['labels']


    initialHist1 = compute_lr_histogram("R", alphabet, images, labels, 20, return_bin_edges=True)
    initialHist2 = compute_lr_histogram("A", alphabet, images, labels, 20, return_bin_edges=True)

    initialMean1 = compute_letter_mean("R", alphabet, images, labels)
    #Image.fromarray(initialMean1, mode='L').save("initial1_mean.png")

    initialMean2 = compute_letter_mean("A", alphabet, images, labels)
    #Image.fromarray(initialMean2, mode='L').save("initial2_mean.png")

    plt.figure()
    plt.title("Letter feature histogram")
    plt.xlabel("LR feature")
    plt.ylabel("# Images")
    histPlot1 = histogram_plot(initialHist1, color='b', alpha=0.75)
    histPlot2 = histogram_plot(initialHist2, color='r', alpha=0.75)
    plt.legend((histPlot1, histPlot2), ("letter 'R'", "letter 'A'"))
    plt.savefig("initials_histograms.png")
    #return plt.bar(range(len(hist_data)), hist_data[0],alpha = alpha, color = color)






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


if __name__ == '__main__':
    A = np.array([[16, 2, 3, 13],
                  [5, 11, 10, 8],
                  [9, 7, 6, 12],
                  [4, 14, 15, 1]])

    B = np.array([[3, 4, 9, 4, 3, 6, 6, 2, 3, 4],
                  [9, 2, 10, 1, 4, 3, 7, 1, 3, 5]])

    output = matrix_manip(A, B)
