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



PLOT = True

class TestPCA(UT.TestCase):
    
    def setUp(self):
        np.random.seed(0)
    
    
    def tearDown(self):
        plt.show()
            
            
    def test_noise_data(self):
        # Samples x dimesnions
        data = np.random.normal(size=(1000, 10))
        filename = "test"
        pca = PCA(data, filename)
        print(pca.components_)
            
            
    def test_baseline_data(self):
        # Samples x dimesnions
        data = self.load_test_rate()
        filename = "test_bs_rate"
        pca = PCA(data, filename)
        print(pca.components_)
        
            
  
    
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