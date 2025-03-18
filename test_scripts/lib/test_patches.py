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

from lib.dopamine import circular_patch, merge_patches

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    nrows = 60

    cpatch = circular_patch(nrows, (30, 20), 2)
    plt.figure()
    plt.imshow(cpatch.reshape((nrows, nrows)), origin="lower")

    cpatch_1 = circular_patch(nrows, (20, 24))
    cpatch_2 = circular_patch(nrows, (20, 20))
    cpatch_1_2 = merge_patches(cpatch_1, cpatch_2)
    plt.figure()
    plt.imshow(cpatch_1_2.reshape((nrows, nrows)), origin="lower")

    cpatch_3 = circular_patch(nrows, (1, 2))
    cpatch_1_2_3 = merge_patches(cpatch_1, cpatch_2, cpatch_3)
    # ~ inverse
    plt.figure()
    plt.imshow(cpatch_1_2_3.reshape((nrows, nrows)), origin="lower")






if __name__ == '__main__':
    main()
