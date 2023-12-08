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
import unittest as UT

from analysis import sequence_correlation as SC

#===============================================================================
# TEST CLASS
#===============================================================================



class TestDBScan(UT.TestCase):

    def setUp(self):
        self.rnd_gen = np.random.default_rng(seed=1)
        self.correlator = SC.SequenceCorrelator(None)



    def tearDown(self):
        plt.show()




#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    UT.main()
