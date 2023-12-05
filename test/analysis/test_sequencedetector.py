#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    dbscan

"""


import unittest as UT

from analysis.sequencedetector import SequenceDetector

import numpy as np
import matplotlib.pyplot as plt

DATA_rows = 70
DATA_WARMUP = 500

R = 2
THRESHOLD = .2
MINIMAL_PEAK_DISTANCE = 12

SINGLE_NEURON = 3466
NEURON_ASSEMBLY = np.arange(3460, 3477)

PLOT = True

class TestSequenceDetector(UT.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.rate = self.load_test_rate()        
        self.sd = SequenceDetector(R, THRESHOLD, MINIMAL_PEAK_DISTANCE)
    
    
    def tearDown(self):
        plt.show()


    def test_number_of_sequences_single_neuron(self):
        self.sd.number_of_sequences(self.rate, SINGLE_NEURON)


    def test_number_of_sequences_many_neurons(self):
        self.sd.number_of_sequences(self.rate, NEURON_ASSEMBLY)

        
    def test_avg_number_of_sequences_many_neurons(self):
        self.sd.number_of_sequences(self.rate, NEURON_ASSEMBLY, avg=True)
    
    
    
    @staticmethod
    def load_test_rate():
        def load_file(filename):
            import pickle
            with open(filename, "rb") as f:
                return pickle.load(f)[0]
            
        rate = load_file("analysis/test/data/rate_Perlin_uniform_baseline.bn")
        rate = rate[:, DATA_WARMUP:] # Warmup of data
        rate = rate[:int(DATA_rows**2)]   # Rows were 70
        return rate
            
    
            
if __name__ == '__main__':
    UT.main()    