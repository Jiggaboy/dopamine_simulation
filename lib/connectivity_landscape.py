# -*- coding: utf-8 -*-
#
# connectivity_landscape.py
# Convention: angles range from -pi to pi
# TODO: Separate the normalization of simplex_noise and perlin
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import numpy as np
import noise

symmetric = 'symmetric' # Connectivity is symmetric around the neuron
homogeneous = 'homogeneous' # preferred directions are all the same
random = 'random' # Preferred directions are random
independent = 'independent' # all-to-all connectivity
perlin = 'perlin'
perlin_uniform = 'perlin_uniform'
simplex_noise = 'simplex_noise'


__all__ = [
    symmetric,
    homogeneous,
    random,
    perlin,
    perlin_uniform,
    simplex_noise,
]


def symmetric(nrow, params={}, **kwargs):
    return


def homogeneous(nrow, params={}, **kwargs):
    phi = params.get('phi', np.pi / 2)

    npop = np.power(nrow, 2)
    return np.ones(npop, dtype=float) * phi


def random(nrow, params={}, bins:int=8):
    # Updated to return angles in v0.1a
    seed = params.get('seed', None)

    np.random.seed(seed)
    npop = np.power(nrow, 2)
    _range = np.linspace(-np.pi, np.pi, bins, endpoint=False)
    direction = np.tile(_range, npop // bins + 1)
    np.random.shuffle(direction)
    return direction[:npop]


def simplex_noise(nrow, params={}, bins:int=8):
    # Updated to return angles in v0.1a
    size = params.get('size', 5)
    base = params.get('base', 0)
    octaves = params.get('octaves', 2)
    persistence = params.get('persistence', 0.5)
    lacunarity = params.get('lacunarity', 2)
    params = {
        "base": base,
        "repeatx": size,
        "repeaty": size,
        "octaves": octaves,
        "persistence": persistence,
        "lacunarity": lacunarity,
    }
    x = y = np.linspace(0, size, nrow, endpoint=False)
    n = [[noise.snoise2(i, j, **params) for j in y] for i in x]
    n = np.asarray(n).ravel()

    directions = np.zeros(shape=n.shape, dtype=float)

    sortindex = np.argsort(n)
    splits = np.array_split(np.arange(nrow**2), bins)
    _range = np.linspace(-np.pi, np.pi, bins, endpoint=False)

    for split, angle in zip(splits, _range):
        directions[sortindex[split]] = angle
    return directions


def perlin(nrow, params={}):
    size = params.get('size', 5)
    base = params.get('base', 0)
    octaves = params.get('octaves', 2)
    persistence = params.get('persistence', 0.5)
    lacunarity = params.get('lacunarity', 2)
    perlin_specs = {
        "base": base,
        "repeatx": size,
        "repeaty": size,
        "octaves": octaves,
        "persistence": persistence,
        "lacunarity": lacunarity,
    }

    x = y = np.linspace(0, size, nrow, endpoint=False)
    n = [[noise.pnoise2(i, j, **perlin_specs) for j in y] for i in x]
    n = np.asarray(n)

    return n.ravel()


def perlin_uniform(nrow, params={}, bins:int=8, *args, **kwargs):
    # Updated to return angles in v0.1a
    """Creates a Perlin configuration and split them into {bins=8} uniform angles."""
    n = perlin(nrow, params, *args, **kwargs)

    directions = np.zeros(shape=n.shape, dtype=float)

    sortindex = np.argsort(n)
    splits = np.array_split(np.arange(nrow**2), bins)
    _range = np.linspace(-np.pi, np.pi, bins, endpoint=False)

    for split, angle in zip(splits, _range):
        directions[sortindex[split]] = angle
    return directions
