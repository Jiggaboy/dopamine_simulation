#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Runs the model with different kinds of configurations.

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
# Copyright 2017 Sebastian Spreizer
# The MIT License
__author__ = ['Sebastian Spreizer', 'Hauke Wernecke']
__contact__ = 'hower@kth.se'
__licence__ = 'MIT License'
__version__ = '0.1'


#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import logging
log = logging.getLogger()

import numpy as np

from lib import pickler as PIC
from lib import functimer
import lib.connection_matrix as cm


# Landscape - Possible values:
#     - random (Random preferred direction, seed:int)
#     - independent (Does not have a preferred direction nor a spatial distribution, seed:int)
#     - symmetric (No pref. direction, seed:int)
#     - Perlin (Pref. direction according to the Perlin noise) {Parameter: size:int, base:int, seed:int}
#     - Perlin_uniform (as for Perlin but euqally distributed throughout the 8 directions)
#     - homogeneous (all have the same preferred direction) {Takes a preferred direction 'phi' with values between 0-7, seed:int}



class ConnectivityMatrix:

    def __init__(self, config):
        log.info("Initialize ConnectivityMatrix…")
        self._rows = config.rows
        self._landscape = config.landscape
        self._path = config.path_to_connectivity_matrix()


    @property
    def connections(self):
        """Full adjacency matrix of exc. and inh. population."""
        # Indegree is the sum of rows?
        # Outdegree is the sum of columns?
        E_EI = np.concatenate((self._EE, self._EI), axis=1)
        I_EI = np.concatenate((self._IE, self._II), axis=1)
        W = np.concatenate((E_EI, I_EI)).T
        return W


    @functimer(logger=log)
    def connect_neurons(self, save:bool=True, save_as_matrix:bool=False, EE_only:bool=False):
        log.info("Connect Neurons…")
        self._EE, self._EI, self._IE, self._II, self.shift = cm.EI_networks(self._landscape, self._rows)
        # self._EE, self.shift = cm.EI_networks(self._landscape, self._rows, EE_only=EE_only)
        log.info("Check for self connection...")
        assert np.all(np.diagonal(self._EE) == 0)
        if not EE_only:
            assert np.all(np.diagonal(self._II) == 0)

        if save:
            log.info(f"Save connectivity matrix object to: {self._path}")
            PIC.save(self._path, self)
            if save_as_matrix:
                log.info(f"Save connectivity matrix (array) to: {self._path}")
                PIC.save_conn_matrix(self._path, self, EE_only=EE_only)


    def load(self, force:bool=False, save:bool=True)->object:
        """
        Loads a cls-instance determined by {self.config}.

        Parameters
        ----------
        save : bool, optional
            Whether to save the object after instantiation (Not if loaded successfully). The default is True.

        Raises
        ------
        FileNotFoundError
            Instantiation of a new object if file not found.

        Returns
        -------
        cls
            Instantiated object.

        """
        log.info(f"Load connectivity matrix from {self._path}…")
        try:
            if force:
                raise FileNotFoundError
            return PIC.load(self._path)
        except (FileNotFoundError, AttributeError):
            self.connect_neurons(save=save)
        return self


    @staticmethod
    def degree(matrix:np.ndarray)->tuple:
        """Returns (indegree, outdegree) of the given matrix."""
        source, target = np.sqrt(matrix.shape).astype(int)
        indegree = matrix.sum(axis=0).reshape((target, target))
        outdegree = matrix.sum(axis=1).reshape((source, source))
        return indegree, outdegree
