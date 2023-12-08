#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:49:52 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt
import class_lib.network_configuration as CN
from class_lib.toroid import Toroid


def perlin_patch(nrows:int, size:int=5, base:int=None):
    rgen = np.random.default_rng()
    base = base or rgen.integers(0, 100)
    p = CN.Perlin_uniform(nrows, size=size, base=base)
    pmax = p.max()
    patchNeurons = p == pmax
    return patchNeurons


def circular_patch(grid:(int, Toroid), center:tuple, radius:float=5, coordinates:np.ndarray=None):
    if not isinstance(grid, Toroid):
        grid = Toroid((grid, grid))

    if coordinates is None:
        side = np.arange(grid.width)
        X, Y = np.meshgrid(side, side)
        xy = np.asarray([X.ravel(), Y.ravel()]).T
    else:
        xy = coordinates

    patch = [grid.get_distance(center, pos, form='squared') <= radius**2 for pos in xy]
    patch = np.asarray(patch)
    return patch


def merge_patches(*patches)->np.ndarray:
    patch = patches[0]
    for p in patches:
        patch = patch | p
    return patch


def plot_patch(nrows:int, patch:np.ndarray):
    plt.figure()
    plt.imshow(patch.reshape((nrows, nrows)), origin="lower")


if __name__ == '__main__':
    nrows = 60
    patch = perlin_patch(nrows)
    plot_patch(nrows, patch)
    cpatch = circular_patch(nrows, (20, 20), 1)
    plot_patch(nrows, cpatch)
    cpatch_1 = circular_patch(nrows, (20, 24))
    cpatch_2 = circular_patch(nrows, (20, 20))
    cpatch_1_2 = merge_patches(cpatch_1, cpatch_2)
    plot_patch(nrows, cpatch_1_2)
    cpatch_3 = circular_patch(nrows, (1, 2))
    cpatch_1_2_3 = merge_patches(cpatch_1, cpatch_2, cpatch_3)
    # ~ inverse
    plot_patch(nrows, ~cpatch_1_2_3)
