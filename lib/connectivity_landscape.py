# -*- coding: utf-8 -*-
#
# connectivity_landscape.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import noise
import matplotlib.pyplot as plt

symmetric = 'symmetric'
homogeneous = 'homogeneous'
random = 'random'
independent = 'independent'
perlin = 'perlin'
perlin_uniform = 'perlin_uniform'





__all__ = [
    symmetric,
    homogeneous,
    random,
    perlin,
    perlin_uniform,
]


SYMMETRIC_LANDSCAPES = (symmetric, independent)


def symmetric(nrow, specs={}):
    return


def homogeneous(nrow, specs={}):
    dir_idx = specs.get('phi', 4)

    npop = np.power(nrow, 2)
    landscape = np.ones(npop, dtype=int) * dir_idx
    return landscape


def random(nrow, specs={}):
    seed = specs.get('seed', 0)

    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.randint(8, size=npop)
    return landscape



def Perlin(nrow, specs={}):
    size = specs.get('size', 5)
    base = specs.get('base', 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size, base=base) for j in y] for i in x]

    # Normalize to the interval [0, 1]
    m = n - np.min(n)
    m /= m.max()
    return m.ravel()


def Perlin_uniform(nrow, specs={}, *args, **kwargs):
    """Creates a Perlin configuration and split them into 8 uniform bins."""
    noise_matrix = Perlin(nrow, specs, *args, **kwargs)

    DIRECTIONS = 8

    a = np.argsort(noise_matrix)
    no_per_direction = np.power(nrow, 2) // DIRECTIONS

    for direction in np.arange(DIRECTIONS):
        idx_of_no_per_direction = a[direction * no_per_direction:(direction + 1) * no_per_direction]
        noise_matrix[idx_of_no_per_direction] = direction

    return noise_matrix.astype(int)


# def move(nrow):
#     return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])
