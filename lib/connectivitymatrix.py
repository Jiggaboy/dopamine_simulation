#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#===============================================================================
# PROGRAM METADATA
#===============================================================================
# Copyright 2017 Sebastian Spreizer
# The MIT License
__author__ = ['Sebastian Spreizer', 'Hauke Wernecke']
__contact__ = 'hower@kth.se'
__licence__ = 'MIT License'
__version__ = '0.3'


#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
from pathlib import Path

import lib.connectivity_landscape as cl
import lib.lcrn_network as lcrn
from class_lib import Group, Connection
from lib import functimer
from lib import pickler as PIC

from lib import SingletonClass

# Landscape - Possible values:
#     - random (Random preferred direction, seed:int)
#     - independent (Does not have a preferred direction nor a spatial distribution, seed:int)
#     - symmetric (No pref. direction, seed:int)
#     - Perlin (Pref. direction according to the Perlin noise) {Parameter: size:int, base:int, seed:int}
#     - Perlin_uniform (as for Perlin but euqally distributed throughout the 8 directions)
#     - homogeneous (all have the same preferred direction) {Takes a preferred direction 'phi' with values between 0-7, seed:int}


class ConnectivityMatrix:

    def __new__(cls, config:object=None, save:bool=True, force:bool=False):
        if config is not None:
            path = config.path_to_connectivity_matrix()
            if not force and PIC.path_exists(path):
                return PIC.load(path)
        return super().__new__(cls)


    def __init__(self, config:object, save:bool=True, **kwargs):
        if config is None:
            return
        logger.info("Initialize ConnectivityMatrix…")
        self._config = config
        self._path = config.path_to_connectivity_matrix()
        self.NE = int(config.rows**2)

        if not hasattr(self, "shift"):
            logger.info("Retrieving the shift...")
            self.shift = self.get_shift(config)
        if not hasattr(self, "_EE"):
            logger.info("Connecting neurons...")
            self.connect_neurons(save=save)

        self.connectivity_matrix = self._weight_synapses(self.connections)
        self.synapses_matrix = self.connections


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


    @functimer(logger=logger)
    def connect_neurons(self, save:bool=True):
        "v0.2: Remove {EE_only} and {save_as_matrix}."
        logger.info("Connect Neurons…")
        self._EE, self._EI, self._IE, self._II = EI_networks(self._config.landscape, self._config.rows, self.shift)
        logger.info("Check for self connection...")
        assert np.all(np.diagonal(self._EE) == 0)
        assert np.all(np.diagonal(self._II) == 0)

        if save:
            logger.info(f"Save connectivity matrix object to: {self._path}")
            PIC.save(self._path, self)


    def get_shift(self, config) -> np.ndarray:
        print("GETTING OLD SHIFT")
        return cl.__dict__[config.landscape.mode](config.rows, config.landscape.params)

    def reset_connectivity_matrix(self)->None:
        self.connectivity_matrix = self._weight_synapses(self.synapses_matrix.copy())


    def _weight_synapses(self, connectivity_matrix:np.ndarray):
        """Weights the synapses according to the exc. and inh. weights, respectively."""
        connectivity_matrix = connectivity_matrix.astype(float)
        connectivity_matrix[:, :self.NE] *= self._config.synapse.exc_weight
        connectivity_matrix[:, self.NE:] *= self._config.synapse.inh_weight
        return connectivity_matrix


    @staticmethod
    def degree(matrix:np.ndarray) -> tuple:
        """Returns (outdegree, indegree) of the given matrix."""
        source, target = np.sqrt(matrix.shape).astype(int)
        outdegree = matrix.sum(axis=0).reshape((target, target))
        indegree = matrix.sum(axis=1).reshape((source, source))
        return outdegree, indegree


class CustomConnectivityMatrix(ConnectivityMatrix):
    def get_shift(self, config):
        shift = cl.__dict__[config.landscape.mode](config.rows, config.landscape.params)
        shift = shift.reshape((config.rows, config.rows))
        shift[30:] = 6
        shift[:30, 45:] = 1
        return np.reshape(shift, config.rows**2)

        print("GET NEW SHIFT")
        params_tmp = config.landscape.params.copy()
        shift = np.zeros((config.rows, config.rows))

        params_tmp["base"] = 21
        shift_bl = cl.__dict__[config.landscape.mode](config.rows // 2, params_tmp)
        shift[:config.rows // 2, :config.rows // 2] = shift_bl.reshape((config.rows // 2, config.rows // 2))

        params_tmp["base"] = 21
        shift_br = cl.__dict__[config.landscape.mode](config.rows // 2, params_tmp)
        shift[:config.rows // 2, config.rows // 2:] = shift_br.reshape((config.rows // 2, config.rows // 2)) + 1

        params_tmp["base"] = 21
        shift_tl = cl.__dict__[config.landscape.mode](config.rows // 2, params_tmp)
        shift[config.rows // 2:, :config.rows // 2] = shift_bl.reshape((config.rows // 2, config.rows // 2)) + 6

        # params_tmp["base"] = 21
        # shift_tr = cl.__dict__[config.landscape.mode](config.rows // 2, params_tmp)
        # shift[config.rows // 2:, config.rows // 2:] = shift_br.reshape((config.rows // 2, config.rows // 2)) + 2

        selectmotif = np.zeros((config.rows // 2, config.rows // 2))
        side = config.rows // 2
        half = side // 2
        selectmotif[:, :half] = 0
        selectmotif[:, half:] = 4

        # main path
        selectmotif[:side, half-3:half]   = 1
        selectmotif[:side, half  :half+3] = 3
        selectmotif[:side, half-1:half+1] = 2

        # intersection
        selectmotif[half:half+2, half-4:half]   = 1
        selectmotif[half:half+2, half-2:half]   = 2
        selectmotif[half:half+2, half:half+4]   = 3
        selectmotif[half:half+2, half:half+2]   = 2

        selectmotif[half+2:half+3, half-5:half]   = 1
        selectmotif[half+2:half+3, half-3:half]   = 2
        selectmotif[half+2:half+3, half-1:half]   = 3
        selectmotif[half+2:half+3, half:half+5]   = 3
        selectmotif[half+2:half+3, half:half+3]   = 2
        selectmotif[half+2:half+3, half:half+1]   = 1

        for i in range(2, 10):
            selectmotif[half+i:half+i+1, max(half-i-3, 0):half]   = 1
            selectmotif[half+i:half+i+1, max(half-i-1, 0):half]   = 2
            selectmotif[half+i:half+i+1, max(half-i+1, 0):half]   = 3
            selectmotif[half+i:half+i+1, max(half-i+3, 0):half]   = 4
            selectmotif[half+i:half+i+1, half:half+i+3]   = 3
            selectmotif[half+i:half+i+1, half:half+i+1]   = 2
            selectmotif[half+i:half+i+1, half:half+i-1]   = 1
            selectmotif[half+i:half+i+1, half:half+i-3]   = 0


        shift[config.rows // 2:, config.rows // 2:] = selectmotif
        # shift_tmp = shift.copy()
        # shift_tmp = np.reshape(shift_tmp, (config.rows, config.rows))
        # shift_tmp = np.roll(shift_tmp, shift=(10, 20), axis=(0, 1))
        # np.reshape(shift_tmp.T, (shift.shape)) + 5
        return np.reshape(shift, config.rows**2) % 8


def EI_networks(landscape, nrowE, shift_matrix:np.ndarray, **kwargs):
    grp_E = Group(nrowE, landscape.stdE)
    grp_I = Group(nrowE // 2, landscape.stdI)

    set_seed(landscape.seed)

    EE_setup = grp_E, grp_E, landscape.is_asymmetric, shift_matrix
    EI_setup = grp_E, grp_I, True
    IE_setup = grp_I, grp_E, True
    II_setup = grp_I, grp_I, False
    setups = (EE_setup, EI_setup, IE_setup, II_setup)

    fct_find_targets = lcrn.independent_targets if landscape.is_independent else lcrn.lcrn_gauss_targets

    @functimer(logger=logger)
    def find_neurons(source, target, allow_self_connections, shift=None):
        # self connections within EE/II are removed anyways. it only is taken into account for finding targets before applying the shift.
        conmat = np.zeros((source.quantity, target.quantity), dtype=np.int16)
        # TODO: Could make sense to draw targets for source.quantitiy, x pos, y pos and only shift for each neuron individually
        for idx in range(source.quantity):
            cnn = Connection(idx, source, target, landscape.connection_probability, allow_self_connections=allow_self_connections)
            direction = None if shift is None else shift_matrix[idx]
            postsynaptic_ids = fct_find_targets(*cnn.get_all(), direction=direction, shift=landscape.shift)

            conmat[idx] = id_to_neuron(postsynaptic_ids, target)
        return conmat


    conmats = []
    for s in setups:
        conmat = find_neurons(*s)
        conmats.append(conmat)
        if kwargs.get("EE_only", False):
            break
    return conmats


def set_seed(seed):
    if seed is None:
        logger.info("Set random seed for landscape generation.")
        np.random.seed()
        seed = np.random.randint(1000)
    logger.info(f"Set seed for landscape generation: {seed}.")
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
