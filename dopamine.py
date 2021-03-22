#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:49:52 2021

@author: hauke
"""

import numpy as np
import matplotlib.pyplot as plt
import custom_class.network_configuration as CN
from custom_class.toroid import Toroid

rgen = np.random.default_rng()

def perlin_patch(nrows:int, size:int=5, base:int=None):
    base = base or rgen.integers(0, 100)
    p = CN.Perlin_uniform(nrows, size=size, base=base)
    pmax = p.max()
    patchNeurons = p == pmax
    return patchNeurons


def circular_patch(nrows:int, center:tuple, radius=5):
    grid = Toroid((nrows, nrows))

    side = np.arange(nrows)
    X, Y = np.meshgrid(side, side)
    xy = np.asarray([X.ravel(), Y.ravel()]).T

    patch = [grid.get_distance(center, pos) < radius for pos in xy]
    patch = np.asarray(patch)
    return patch


def plot_patch(nrows:int, patch:np.ndarray):
    plt.figure()
    plt.imshow(patch.reshape((nrows, nrows)), origin="lower")


if __name__ == '__main__':
    nrows = 60
    patch = perlin_patch(nrows)
    plot_patch(nrows, patch)
    cpatch = circular_patch(nrows, (20, 20))
    plot_patch(nrows, cpatch)
