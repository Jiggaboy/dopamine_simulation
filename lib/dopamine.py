#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

import numpy as np
import matplotlib.pyplot as plt

from class_lib.toroid import Toroid


def circular_patch(grid:(int, Toroid), center:tuple, radius:float=5):
    if not isinstance(grid, Toroid):
        grid = Toroid((grid, grid))

    patch = np.roll(grid.get_distances(), center, axis=(1, 0)) <= radius
    return patch.flatten()


def merge_patches(*patches)->np.ndarray:
    patch = patches[0]
    for p in patches:
        patch = patch | p
    return patch


if __name__ == '__main__':
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
