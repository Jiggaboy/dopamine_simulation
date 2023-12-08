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

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable
from lib import universal as UNI
import unittest as UT

from analysis import sequence_correlation as SC
from params import BaseConfig

#===============================================================================
# TEST CLASS
#===============================================================================


test_rows = 40


class TestDBScan(UT.TestCase):

    def setUp(self):
        self.rnd_gen = np.random.default_rng(seed=1)
        cfg = BaseConfig()
        cfg.rows = test_rows
        self.correlator = SC.SequenceCorrelator(cfg)


    def tearDown(self):
        plt.show()


    def test_has_spikes_at_center(self):
        # has_spikes_at_center(spikes_in_sequence:np.ndarray, center:tuple, coordinates:np.ndarray) -> bool:#
        center = (10, 10)
        coordinates = UNI.get_coordinates(nrows=test_rows)

        spikes_is_in_sequence = np.asarray([
            [0, 10, 10],
        ])

        spikes_is_not_in_sequence = np.asarray([
            [10, 0, 10],
        ])

        spikes_mixed = np.asarray([
            [0, 10, 10],
            [10, 0, 10],
        ])

        self.assertTrue(self.correlator.has_spikes_at_center(spikes_is_in_sequence, center, coordinates))
        self.assertFalse(self.correlator.has_spikes_at_center(spikes_is_not_in_sequence, center, coordinates))
        self.assertTrue(self.correlator.has_spikes_at_center(spikes_mixed, center, coordinates))


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    UT.main()
