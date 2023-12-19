#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable
import unittest

from class_lib import Population
from params import TestConfig

#===============================================================================
# TEST
#===============================================================================




class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pop = Population(TestConfig())


    @classmethod
    def tearDownClass(cls):
        cls.pop.plot_population()
        cls.pop.hist_in_degree()
        cls.pop.hist_out_degree()
        print()
        print(f"Average activation: {cls.pop.connectivity_matrix.mean()}")


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_no_self_connection(self):
        for idx, n in enumerate(self.pop.connectivity_matrix):
            self.assertFalse(n[idx], msg=f"{idx} has a self connection!")


    def test_connectivity(self):
        self.assertGreater(self.pop.connectivity_matrix.nonzero()[0].size, self.pop.neurons.size)

        for axon in self.pop.connectivity_matrix.T:
            synapses = axon[axon != 0]
            excitatory = all(synapses > 0)
            inhibitory = all(synapses < 0)
            self.assertTrue(excitatory or inhibitory)


    # def test_cap_of_synaptic_updates(self):
    #     self.assertTrue(hasattr(self.pop, "connectivity_cap"))
    #     connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
    #     self.assertTrue(np.all(connectivity_diff >= 0))
    #     synapses = np.arange(10, 30)
    #     for _ in range(20):
    #         self.pop.update_synaptic_weights(synapses, learning_rate=.01)
    #     connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
    #     self.assertTrue(np.all(connectivity_diff >= 0))


    def test_plot_indegree(self):
        self.assertTrue(hasattr(self.pop, "plot_indegree"), "No plot_indegree attribute...")
        self.pop.plot_indegree()


    def _test_plot_gradient(self):
        self.assertTrue(hasattr(self.pop, "plot_gradient"), "No plot_gradient attribute...")
        self.pop.plot_gradient()




if __name__ == '__main__':
    from params import TestConfig
    unittest.main()
