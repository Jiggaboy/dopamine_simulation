#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:00:36 2021

@author: hauke
"""

import numpy as np
from collections import namedtuple

import configuration as CF

from custom_class.toroid import Toroid
from custom_class.neurontype import NeuronType

Distribution = namedtuple("Distribution", ("sigma", "multiplier", "steepness"))


# Gaussian parameter
SIGMA_EXC = 7.
SIGMA_INH = 10

CONNECTION_PROBABILITY = 0.1

torus = Toroid((20, 20))
# torus = Toroid((CF.SPACE_WIDTH, CF.SPACE_HEIGHT))

# GRID 60x60, prob.=10%
exc_dist = Distribution(SIGMA_EXC, multiplier=1, steepness=1)
inh_dist = Distribution(SIGMA_INH, multiplier=1, steepness=1)

# # GRID 50x50, prob.=10%
# exc_dist = Distribution(SIGMA_EXC, multiplier=24, steepness=0.00008)
# inh_dist = Distribution(SIGMA_INH, multiplier=24, steepness=0.00008)

# GRID 5x5
# exc_dist = Distribution(SIGMA_EXC, multiplier=5, steepness=.01)
# inh_dist = Distribution(SIGMA_INH, multiplier=5, steepness=.01)

# # GRID 35x35
# exc_dist = Distribution(SIGMA_EXC, multiplier=50, steepness=.0016)
# inh_dist = Distribution(SIGMA_INH, multiplier=50, steepness=.0016)



def get_distribution(type_:NeuronType)->dict:
    if type_ == NeuronType.EXCITATORY:
        distribution = exc_dist
    elif type_ == NeuronType.INHIBITORY:
        distribution = inh_dist

    factor = calc_factor(distribution.sigma, distribution.multiplier, CONNECTION_PROBABILITY)

    gauss = {}
    for w in range(torus.width):
        for h in range(torus.height):
            distance = torus.get_distance((w, h), form="squared")
            gauss[distance] = draw_from_gaussian_distribution(distance,
                                                              factor=factor,
                                                              sigma=distribution.sigma,
                                                              steep=distribution.steepness)
    return gauss


def get_prefactor(sigma:float)->float:
    prefactor = 1 / np.sqrt(2 * np.pi)
    return prefactor / np.power(sigma, 1)


def calc_factor(sigma:float, multplier:float, probability:float)->float:
    return get_prefactor(sigma) * multplier * probability


def draw_from_gaussian_distribution(x:float, factor:float, mu:float=0.0, sigma:float=1.0, steep:float=0.5)->float:
    return factor * np.exp(- steep * np.power((x - mu) / sigma, 2))



import unittest
import matplotlib.pyplot as plt

class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.exc_distribution = get_distribution(NeuronType.EXCITATORY)
        cls.inh_distribution = get_distribution(NeuronType.INHIBITORY)
        cls.distributions = [cls.exc_distribution,
                             cls.inh_distribution,]


    @classmethod
    def tearDownClass(cls):
        plt.figure()
        for d in cls.distributions:
            xy = np.asarray(sorted(d.items()))
            x = np.sqrt(xy[:, 0])
            y = xy[:, 1]
            plt.plot(x, y)
            plt.xlim(left=0)

        xy_e = np.asarray(sorted(cls.distributions[0].items()))
        xy_i = np.asarray(sorted(cls.distributions[1].items()))
        x = xy_e[:, 0]
        y = xy_e[:, 1] - xy_i[:, 1]
        plt.plot(x, y)
        plt.axhline()
        plt.xlim(left=0)




    def test_maximum_probability(self):
        for d in self.distributions:
            self.assertLessEqual(d[0] , 1)





if __name__ == '__main__':
    unittest.main()
