# -*- coding: utf-8 -*-
#
# connectivity_landscape.py

import numpy as np
import noise

symmetric = 'symmetric' # Connectivity is symmetric around the neuron
homogeneous = 'homogeneous' # preferred directions are all the same
random = 'random' # Preferred directions are random
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


SYMMETRIC_LANDSCAPES = (symmetric)


def symmetric(nrow, specs={}):
    return


def homogeneous(nrow, specs={}):
    dir_idx = specs.get('phi', 4)

    npop = np.power(nrow, 2)
    landscape = np.ones(npop, dtype=int) * dir_idx
    return landscape


def random(nrow, specs={}):
    seed = specs.get('seed', None)

    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.randint(8, size=npop)
    return landscape


def simplex_noise(nrow, params={}, directions:int=8):
    size = params.get('size', 5)
    base = params.get('base', 0)
    octaves = params.get('octaves', 2)
    persistence = params.get('persistence', 0.5)
    lacunarity = params.get('lacunarity', 2)
    specs = {
        "base": base,
        "repeatx": size,
        "repeaty": size,
        "octaves": octaves,
        "persistence": persistence,
        "lacunarity": lacunarity,
    }
    x = y = np.linspace(0, size, nrow, endpoint=False)
    n = [[noise.snoise2(i, j, **specs) for j in y] for i in x]
    n = np.asarray(n).ravel()
    max_distance = n.max() - n.min()
    n = (n * 1.) % max_distance

    direction_matrix = np.zeros(shape=n.shape, dtype=int)

    a = np.argsort(n)
    no_per_direction = np.power(nrow, 2) // directions

    # Binning into the directions
    for direction in np.arange(directions):
        # Find these index which correspond to the (lowest) quantile and assign the direction 0 to it.
        idx_of_no_per_direction = a[direction * no_per_direction:(direction + 1) * no_per_direction]
        direction_matrix[idx_of_no_per_direction] = direction
    return direction_matrix




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
    max_distance = n.max() - n.min()
    max_distance = n.max() - n.min()
    n = (n * 1) % max_distance

    # Normalize to the interval [0, 1]
    m = n - np.min(n)
    m /= m.max()
    return m.ravel()


def perlin_uniform(nrow, specs={}, directions:int=8, *args, **kwargs):
    """Creates a Perlin configuration and split them into 8 uniform bins."""
    noise_matrix = perlin(nrow, specs, *args, **kwargs)
    direction_matrix = np.zeros(shape=noise_matrix.shape, dtype=int)

    a = np.argsort(noise_matrix)
    no_per_direction = np.power(nrow, 2) // directions

    # Binning into the directions
    for direction in np.arange(directions):
        # Find these index which correspond to the (lowest) quantile and assign the direction 0 to it.
        idx_of_no_per_direction = a[direction * no_per_direction:(direction + 1) * no_per_direction]
        direction_matrix[idx_of_no_per_direction] = direction
    return direction_matrix
