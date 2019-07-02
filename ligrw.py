#!/usr/bin/env python
#coding:utf-8

"""
Purpose: 
    code for "A New Outlier Detection Model Using Random Walk on Local Information Graph, IEEE Access 2018"
    
Authors: 
    Chao -- < chaosimpler@gmail.com >

License: 
    GNU GPL v3

Created: 
    07/02/19
"""

from __future__ import division
import logging
import numpy as np
import pandas as pd


#----------------------------------------------------------------------
def scale_to_range(arr_data, f_range = (0, 1)):
    r""" scale the data to the specified range
    """
    
    min_value = np.min(arr_data)
    max_value = np.max(arr_data)        
    
    new_min = f_range[0]
    new_max = f_range[1]
    
    smooth_factor = 1e-13
    if len(arr_data.shape) >= 3:
        raise ValueError("This function could only process the 1-dim or 2-dim array data.")
    if (new_max < new_min):
        raise ValueError("scale range set error.")
    
    if max_value != min_value:
        new_data = (new_max - new_min) * (arr_data - arr_data.min(axis = 0)) / \
            (arr_data.max(axis = 0) - arr_data.min(axis = 0) + smooth_factor) + new_min
    else:
        new_data = np.zeros_like(arr_data) + new_min

    return new_data


#----------------------------------------------------------------------

def calc_sd_vector(trans_mat,
                   num_simulations = 10000,
                   eps = 1e-10,
                   restart_vector = None,
                   damping_factor = 0.15):
    """ calculate the stationary distribution vector by using Power Iteration
    
    Args:
        trans_mat (ndarray, shape=(sample_size,sample_size)): transition matrix;
        num_simulations (int, optional): max iteration number, default = 10000；
        eps (float, optional): convergence threshold, default = 1e-10;
        restart_vector (array, shape=(sample_size,), optional): restart vector, default = None;
        damping_factor (float, optional): damping factor, default = 0.15;
    Returns :
        arr_vect (array, shape=(sample_size,)): visiting probabilitiy of each sample after the
          random walk process reaches equilibrium.
    """
    
    sample_size = trans_mat.shape[0]
    
    # init distribution vector
    arr_dvect = np.array([1. / sample_size for _ in range(sample_size)])
    
    if not restart_vector is None:
        if len(restart_vector) != sample_size:
            raise ValueError("Restart vector must has equal length with sample size.")
    
    converged = False
    
    for i in range(num_simulations):
        
        if restart_vector is None:
            # without restart process
            arr_dv1 = np.dot(trans_mat, arr_dvect)
        else:
            # with restart process
            arr_dv1 = damping_factor *  restart_vector + (1 - damping_factor) * np.dot(trans_mat, arr_dvect)

        # calculate the norm
        dv1_norm = np.linalg.norm(arr_dv1)
        # re-normalize the vector
        arr_new_dvect = arr_dv1 / dv1_norm
        
        iter_eps = np.linalg.norm((arr_new_dvect - arr_dvect), ord= 1)
        if  iter_eps < eps:
            converged = True
            break
        
        arr_dvect = arr_new_dvect
    
    return arr_dvect

#----------------------------------------------------------------------
def calc_similarity_matrix(data_mat):
    r""" calculate the similarity matrix of the dataset.
    """
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from random import sample

    (sample_size, dim_size) = data_mat.shape
    
    EXP_THESHOLD = 20
    SAMPLE_RATIO = 0.5   
    
    # distance matrix
    pairwise_dists = squareform(pdist(data_mat, 'euclidean'))
    

    # estimate the bandwidth 
    P = int(sample_size * SAMPLE_RATIO)
    if P == 0:
        msg = "Sample ratio is too small which cause the sample count"\
            "equals to zero!"
        raise ValueError(msg)

    abs_sum = 0.0
    for i in range(P):
        x1, x2 = sample(range(sample_size), 2)
        abs_sum += np.linalg.norm(data_mat[x1, :] - data_mat[x2, :], ord= 2) ** 2
    bw = np.sqrt(abs_sum / P )
    
    
    # calculate the similarity matrix
    xmat = pairwise_dists ** 2 / (2 * bw ** 2)
    smat = np.zeros_like(xmat)
    idx = np.where(xmat < EXP_THESHOLD)
    smat[idx] = np.exp(-xmat[idx])

    return smat

#----------------------------------------------------------------------
def calc_transition_matrix(sim_mat):
    r""" calculate the transition matrix from the similarity matrix
    """

    # normalize the similarity matrix by column to
    # generate the transition matrix
    a = sim_mat
    b = np.sum(sim_mat, axis= 0)    # sum by column
    trans_mat = np.divide(a, b, out = np.zeros_like(a), where = b != 0)
    return trans_mat

#----------------------------------------------------------------------
def calc_knn_graph(arr_sim_mat, k):
    """ a zero-one matrix which helps to calculate the similarity matrix of local information graph
    """

    (sample_size,dim_size)=arr_sim_mat.shape

    if k >= sample_size:
        k = sample_size
        
    sorted_sim_mat = np.sort(arr_sim_mat, axis=1)
    arr_k_radius = sorted_sim_mat[:,-k]
    arr_knn_graph =  (arr_sim_mat.T >= arr_k_radius).astype(int).T

    return arr_knn_graph


#----------------------------------------------------------------------
def calc_restart_vector(sim_mat, rv_type):
    r""" calculate the restart vector

    Args:
        sim_mat (array): similarity matrix
        rv_type (int): restart vector type (1 = global, 2 = local)
    Returns:
        arr_rv (array): an array contains the restart probability of each object;
    """

    trans_mat = calc_transition_matrix(sim_mat)

    if rv_type == 1:

        # global restart vector
        gvp = calc_sd_vector(trans_mat)
        gvp1 = scale_to_range(-1 * gvp, f_range= (1, 10))
        arr_rv = gvp1 / gvp1.sum()

    elif rv_type == 2:

        # local restart vector
        arr_most_sim = np.sort(sim_mat, axis= 1)[:, -2]
        arr_wsim = scale_to_range(-1 * arr_most_sim, f_range= (1, 10))
        arr_rv = arr_wsim / arr_wsim.sum()    
    else:
        raise Exception("Unknown Restart vector type = {}.".format(rv_type))

    return arr_rv




#----------------------------------------------------------------------
def calc_outlier_score(data_mat, k, damping_factor = 0.15, rv_type = 1, max_iter_num = 10000):
    r""" calculate the outlier score by using LIGRW
        
    Args:
        data_mat (ndarray,shape=(sample_size,dim_size)): data matrix (rows =  samples, columns = attributes);
        k (int): neighbor count;
        damping_factor (float, optional): damping factor (default = 0.15);
        rv_type (int, optional): type of restart vector (1 = global, 2 = local)；
        max_iter_num (int,optional): max iteration count (default = 10000);
    Returns:
        ndarray (shape=(sample_size,)): an array contains the outlier score of each object;
    """

    (sample_size,dim_size) = data_mat.shape

    # calculate the similarity matrix
    sim_mat = calc_similarity_matrix(data_mat)

    # construct the zero-one matrix to calculate the
    # similarity matrix of local information graph
    knn_graph = calc_knn_graph(sim_mat, k)

    
    # calculate the similarity matrix of local information graph
    LIG_sim_mat = sim_mat * knn_graph
    LIG_trans_mat = calc_transition_matrix(LIG_sim_mat)


    # construct the restart vector
    arr_rv = calc_restart_vector(sim_mat, rv_type)      

    # calculate the outlier score
    arr_score = calc_sd_vector(LIG_trans_mat, 
                               restart_vector = arr_rv,
                               damping_factor = damping_factor)

    return arr_score