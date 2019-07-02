#!/usr/bin/env python
#coding:utf-8

"""
Purpose: 
    Demonstrate how to use the LIGRW algorithm
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
def gen_synthetic_dataset():
    r""" generate a synthetic dataset
    """

    # generate cluster 1
    data_mat1 = np.random.uniform([1, 1], [2, 2],(50, 2))
    
    # generate cluster 2
    sample_count = 100
    center = [8, 5]
    radius = 1.0
    rho = np.sqrt(np.random.uniform(0, 1, sample_count))
    phi = np.random.uniform(0, 2*np.pi, sample_count)
    x = radius * rho * np.cos(phi) + center[0]
    y = radius * rho * np.sin(phi) + center[1]
    data_mat2 = np.vstack((x, y)).T        
    
    # outliers
    data_outlier = np.array([[8, 6.5],[6, 1]])
    
    data_mat = np.vstack((data_mat1, data_mat2, data_outlier))
    
    return data_mat
    
    

if __name__ == '__main__':
    
    from ligrw import calc_outlier_score
    
    # generate a synthetic dataset
    data_mat = gen_synthetic_dataset()
    
    # calculate outlier score using LIGRW with Global restart vector
    arr_os_global = calc_outlier_score(data_mat, k = 8, rv_type= 1)
    
    # calculate outlier score using LIGRW with Local restart vector
    arr_os_local = calc_outlier_score(data_mat, k = 8, rv_type= 2)

    
    print 'over'
    
    
