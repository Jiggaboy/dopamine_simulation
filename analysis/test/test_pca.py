#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    pca

"""


import unittest as UT

from ..pca import PCA

import numpy as np
import matplotlib.pyplot as plt

from util import pickler as PIC
import cflogger
logger = cflogger.getLogger()



PLOT = True

class TestPCA(UT.TestCase):
    
    def setUp(self):
        np.random.seed(0)
    
    
    def tearDown(self):
        plt.show()
            
            
    def test_noise_data(self):
        # Samples x dimesnions
        data = np.random.normal(size=(1000, 2), loc=2.)
        data[:, 0] *= 5
        plt.figure("raw data")
        plt.scatter(*data.T)
        plt.xlim(-5, 30)
        plt.ylim(-5, 30)
        
        filename = "test"
        pca = PCA(data, filename, force=True)
        
        for i in range(2):
            logger.info(pca.components_[i] * pca.explained_variance_[i])
            plt.arrow(*data.mean(axis=0), *pca.components_[i] * np.sqrt(pca.explained_variance_[i]), head_width=1)
            
        def sv_of_covariance_matrix(cov:np.ndarray):
            s = np.linalg.svd(cov, full_matrices=True, compute_uv=False)
            return s
        
        cov = pca.get_covariance()
        
        logger.info(f"Explained variance by PCA: {pca.explained_variance_}")
        logger.info(f"Singular values by SVD: {sv_of_covariance_matrix(cov)}")
        logger.info(f"Singular values by PCA: {pca.singular_values_}")
        logger.info(f"Singular values by PCA: {pca.singular_values_ ** 2 / 1000}")
        logger.info(pca.n_samples_)
        
        
            
        #logger.info(pca.components_)
        #logger.info(pca.explained_variance_)
        #transformed_components = pca.transform(pca.components_)
            
            
    def test_baseline_data(self):
        # Samples x features
        data = self.load_test_rate()
        filename = "test_bs_rate"
        pca = PCA(data.T, filename)
        logger.info(pca.components_)
        
            
  
    
    @staticmethod
    def load_test_rate():
        def load_file(filename):
            import pickle
            with open(filename, "rb") as f:
                return pickle.load(f)[0]
            
        rate = load_file("analysis/test/data/rate_Perlin_uniform_baseline.bn")
        rate = rate[:int(70**2)]   # Rows were 70
        return rate
            
            
    
            
if __name__ == '__main__':
    UT.main()