#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-22

@author: Hauke Wernecke

"""
import cflogger
logger = cflogger.getLogger()


import numpy as np
import sklearn.decomposition as sk

import util.pickler as PIC


def PCA(data:np.ndarray, filename:str=None, force:bool=False, **save_kwargs)->object:
    """
    Loads a PCA object if possible.
    If forced or unable to load a PCA object, create new and run PCA before saving it.
    
    
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    try:
        if force:
            raise FileNotFoundError
        pca = PIC.load_pca(filename)
        logger.info("Loaded PCA...")
    except (FileNotFoundError, TypeError):
        logger.info(f"PCA of data with shape {data.shape} (expects n_samples x n_features)...")
        pca = sk.PCA(svd_solver='full')
        # pca.fit: data has shape -> n_samples x n_features
        pca.fit(data)
        if filename is not None:
            PIC.save_pca(pca, filename, **save_kwargs)
    return pca


def sum_variances(explained_variance_ratio:np.ndarray)->tuple:
    "numpy.cumsum"
    cumRatio = []
    for idx, el in enumerate(explained_variance_ratio):
        try:
            cumRatio.append(cumRatio[idx-1] + el)
        except IndexError:
            cumRatio.append(el)
    cumRatio = np.asarray(cumRatio)
    xRange = range(1, len(cumRatio)+1)

    return (xRange, cumRatio)