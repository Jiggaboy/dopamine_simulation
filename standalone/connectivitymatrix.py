#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Creates a connectivity matrix based on a landscape and network parameters.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
# Copyright 2017 Sebastian Spreizer
# The MIT License
__author__ = ['Sebastian Spreizer', 'Hauke Wernecke']
__contact__ = 'hower@kth.se'
__licence__ = 'MIT License'
__version__ = '0.4'


#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import numpy as np
import matplotlib.pyplot as plt

import util.landscape  as ls
import util.connectivity_landscape as cl
import util.lcrn_network as lcrn
# from lib import pickler as PIC


DIRECTIONS = 8

def main():
    rows = 50


    # landscape = ls.Landscape("symmetric", stdE=2.75, stdI=3., shift=1., connection_probability=.375, seed=0)
    # cm = ConnectivityMatrix(rows=rows, landscape=landscape)
    # print(cm.connections)

    landscape = ls.Landscape("simplex_noise", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
                            params={"size": 2.5, "base": 0, "octaves": 2, "persistence": .5,}, seed=0)
    cm = ConnectivityMatrix(rows=rows, landscape=landscape)
    print(cm.connections)
    print(cm.shift)

    # landscape = ls.Landscape("random", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
    #                         params={"size": 2.5, "base": 0, "octaves": 2, "persistence": .5,}, seed=0)
    # cm = ConnectivityMatrix(rows=rows, landscape=landscape)
    # print(cm.connections)

    # landscape = ls.Landscape("homogeneous", stdE=2.75, stdI=3., shift=1., connection_probability=.375,
    #                         params={"size": 2.5, "base": 0, "octaves": 2, "persistence": .5,}, seed=0)
    # cm = ConnectivityMatrix(rows=rows, landscape=landscape)
    # print(cm.connections)

    outdegree, indegree = cm.degree(cm.connections[:rows**2, :rows**2])
    plt.figure()
    plt.imshow(indegree.reshape((rows, rows)), origin="lower")

    plt.figure()
    plt.imshow(cm.connections[:rows**2, :rows**2])
    plt.show()


# Landscape - Possible values:
#     - random (Random preferred direction, seed:int)
#     - independent (Does not have a preferred direction nor a spatial distribution, seed:int)
#     - symmetric (No pref. direction, seed:int)
#     - simplex_noise (Pref. direction according to the Simplex noise) {Parameter: size:int, base:int, seed:int}
#     - homogeneous (all have the same preferred direction) {Takes a preferred direction 'phi' with values between 0-7, seed:int}


class ConnectivityMatrix:

    # def __new__(cls, config:object=None, save:bool=True, force:bool=False):
        # if config is not None:
        #     path = config.path_to_connectivity_matrix()
            # if not force and PIC.path_exists(path):
                # return PIC.load(path)
        # return super().__new__(cls)


    # def __init__(self, config:object, save:bool=True, **kwargs):
    def __init__(self, rows:int, landscape:object, **kwargs):
        self.rows = rows
        self.landscape = landscape
        self.NE = int(rows**2)


        # logger.info("Initialize ConnectivityMatrix…")
        # self._config = config
        # self._path = config.path_to_connectivity_matrix()
        # self.NE = int(config.rows**2)


        # This is only done if the connectivity matrix is not loaded
        if not hasattr(self, "shift"):
            self.shift = cl.__dict__[landscape.mode](rows, landscape.params)
        if not hasattr(self, "_EE"):
            # logger.info("Connecting neurons...")
            # self.connect_neurons(save=save)
            self.connect_neurons()


    @property
    def EE_connections(self):
        return self._EE

    @property
    def II_connections(self):
        return self._II

    @property
    def IE_connections(self):
        # Target-source notation
        return self._IE

    @property
    def EI_connections(self):
        # Target-source notation
        return self._EI


    @property
    def connections(self):
        """Full adjacency matrix of exc. and inh. population."""
        # Indegree is the sum of rows?
        # Outdegree is the sum of columns?
        E_EI = np.concatenate((self._EE, self._EI), axis=1)
        I_EI = np.concatenate((self._IE, self._II), axis=1)
        W = np.concatenate((E_EI, I_EI)).T
        return W


    # @functimer(logger=logger)
    def connect_neurons(self, save:bool=True):
        "v0.2: Remove {EE_only} and {save_as_matrix}."
        # logger.info("Connect Neurons…")
        self._EE, self._EI, self._IE, self._II = EI_networks(self.landscape, self.rows, self.shift)
        # logger.info("Check for self connection...")
        assert np.all(np.diagonal(self._EE) == 0)
        assert np.all(np.diagonal(self._II) == 0)

        # if save:
        #     logger.info(f"Save connectivity matrix object to: {self._path}")
            # PIC.save(self._path, self)


    @staticmethod
    def degree(matrix:np.ndarray) -> tuple:
        """Returns (outdegree, indegree) of the given matrix."""
        source, target = np.sqrt(matrix.shape).astype(int)
        outdegree = matrix.sum(axis=0).reshape((target, target))
        indegree = matrix.sum(axis=1).reshape((source, source))
        return outdegree, indegree


def EI_networks(landscape, nrowE, shift_matrix:np.ndarray, **kwargs):
    grp_E = Group(nrowE, landscape.stdE)
    grp_I = Group(nrowE // 2, landscape.stdI)

    set_seed(landscape.seed)

    EE_setup = grp_E, grp_E, landscape.is_asymmetric, shift_matrix
    EI_setup = grp_E, grp_I, True
    IE_setup = grp_I, grp_E, True
    II_setup = grp_I, grp_I, False
    setups = (EE_setup, EI_setup, IE_setup, II_setup)

    # @functimer(logger=logger)
    def find_neurons(source, target, allow_self_connections, shift=None):
        # self connections within EE/II are removed anyways. it only is taken into account for finding targets before applying the shift.
        conmat = np.zeros((source.quantity, target.quantity), dtype=np.int16)
        # TODO: Could make sense to draw targets for source.quantitiy, x pos, y pos and only shift for each neuron individually
        for idx in range(source.quantity):
            cnn = Connection(idx, source, target, landscape.connection_probability, allow_self_connections=allow_self_connections)
            direction = None if shift is None else shift_matrix[idx]
            postsynaptic_ids = lcrn.lcrn_gauss_targets(*cnn.get_all(), direction=direction, shift=landscape.shift)

            conmat[idx] = id_to_neuron(postsynaptic_ids, target)
        return conmat


    conmats = []
    for s in setups:
        conmat = find_neurons(*s)
        conmats.append(conmat)
        if kwargs.get("EE_only", False):
            break
    return conmats

#===============================================================================
# DATACLASSES
#===============================================================================
from dataclasses import dataclass

@dataclass
class Group:
    rows: int
    std: float

    @property
    def quantity(self):
        return self.rows ** 2


@dataclass
class Connection:
    idx: int
    source: Group
    target: Group
    connection_probability: float
    allow_self_connections: bool = None

    @property
    def no_of_targets(self)->int:
        return int(self.connection_probability * self.target.quantity)

    @property
    def std(self)->float:
        return self.source.std


    def get_all(self):
        return self.idx, self.source.rows, self.target.rows, self.no_of_targets, self.std, self.allow_self_connections



#===============================================================================
# PLOTTING
#===============================================================================


def calculate_direction(x, bins=DIRECTIONS, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v


def plot_shift(X=None, Y=None, D=None, name:str=None, **kwargs):
    U, V = calculate_direction(D, **kwargs)
    plt.quiver(X, Y, U, V, pivot='middle', scale_units="xy", scale=1.25, units="dots", width=2)


def plot_shift_arrows(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    X, Y = np.meshgrid(np.arange(source), np.arange(source))

    plot_shift(X, Y, shift)


#===============================================================================
# UTIL
#===============================================================================


def set_seed(seed):
    if seed is None:
        # logger.info("Set random seed for landscape generation.")
        np.random.seed()
        seed = np.random.randint(1000)
    # logger.info(f"Set seed for landscape generation: {seed}.")
    np.random.seed(seed)


def id_to_neuron(target_id:np.ndarray, target_grp:Group) -> np.ndarray:
    """
    Counts the occurrences of the targets.

    v0.2: Refactor to bincount. Removal of target_grp parameter.

    Parameters
    ----------
    target_id : np.ndarray
        Array of targets. Must be an int-type array.
    target_grp : Group
        Target group used to map the targets to the right shape.

    Returns
    -------
    np.ndarray
        Count across targets.

    """
    return np.bincount(target_id, minlength=target_grp.quantity)


if __name__ == '__main__':
    main()
