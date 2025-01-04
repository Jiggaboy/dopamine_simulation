#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Plotting the histogram of indegrees (with and without patch).
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
import pandas as pd

import lib.dopamine as DOP
import lib.universal as UNI
from lib.universal import get_neurons_from_patch
from lib.connectivitymatrix import ConnectivityMatrix
from params import config

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    conn = ConnectivityMatrix(config).load()
    radius = config.radius[0]
    amount = config.AMOUNT_NEURONS[0]

    _, indegree = conn.degree(conn.connections[:config.rows**2, :config.rows**2])

    indegrees = pd.DataFrame(indegree.flatten(), columns=["baseline"])

    for name, center in config.center_range.items():
        for percent in UNI.make_iterable(config.PERCENTAGES):
            print(name, center)
            patch = DOP.circular_patch(config.rows, center, float(radius))
            dop_patch = get_neurons_from_patch(patch, amount)
            patchy_indegree = indegree.flatten()
            patchy_indegree[dop_patch] *= (1 + percent)
            id_ = f"{name} {percent}"
            indegrees[id_] = patchy_indegree

    plt.figure()
    plt.title(f"Radius: {radius}")
    plt.hist(indegrees, bins=25, label=indegrees.columns)
    plt.legend()
    plt.show()

#===============================================================================
# METHODS
#===============================================================================






if __name__ == '__main__':
    main()
