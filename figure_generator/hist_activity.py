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

from params import config
import lib.pickler as PIC

N = 11
THRESHOLD = .4

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    for tag in config.baseline_tags:
        rate = PIC.load_rate(tag, skip_warmup=True, exc_only=True, config=config, sub_directory=config.sub_dir)
        bins = np.linspace(0, 1, N, endpoint=True)
        H, edges = np.histogram(rate.ravel(), density=True, bins=bins)
        portion = H[edges[:-1] >= THRESHOLD].sum() / H.sum()
        plt.bar(edges[:-1], H, width=bins[1], align="edge")
        print(portion)

#===============================================================================
# METHODS
#===============================================================================



if __name__ == '__main__':
    main()
