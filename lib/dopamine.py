#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

import numpy as np
from functools import cache

from class_lib.toroid import Toroid


@cache
def circular_patch(grid:(int, Toroid), center:tuple, radius:float=5):
    if not isinstance(grid, Toroid):
        grid = Toroid((grid, grid))

    patch = np.roll(grid.get_distances(), center, axis=(1, 0)) <= radius
    return patch.flatten()


@cache
def merge_patches(*patches)->np.ndarray:
    patch = patches[0]
    for p in patches:
        patch = patch | p
    return patch
