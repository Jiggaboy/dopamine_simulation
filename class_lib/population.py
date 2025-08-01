#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
History:
    v0.2: Remove update_synaptic_weights. Remove grid and coordinates, and _init_neurons (incl. the member inh_neurons), and __len__.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from cflogger import logger

import numpy as np

# from class_lib.toroid import Toroid
from lib.connectivitymatrix import ConnectivityMatrix
# import lib.universal as UNI


class Population:

    def __init__(self, config, save:bool=True, force:bool=False):
        super().__init__()
        logger.info("Create new Population…")
        self._config = config
        self._landscape = config.landscape
        self._synapse = config.synapse

        # self._init_neurons(self._config.rows)
        self.NE = int(config.rows**2)

        # self.grid = Toroid(self._config.rows)
        # self.coordinates = self.populate_grid()

        self.connectivity_matrix, self.synapses_matrix, self.shift = self.set_up_neuronal_connections(save, force)


    # def __len__(self)->int:
    #     return len(self.neurons)


    @property
    def EE_connections(self):
        return self.connectivity_matrix[:self.NE, :self.NE]


    @property
    def II_connections(self):
        return self.connectivity_matrix[self.NE:, self.NE:]


    @property
    def IE_connections(self):
        # Target: source notation
        return self.connectivity_matrix[self.NE:, :self.NE]


    @property
    def EI_connections(self):
        return self.connectivity_matrix[:self.NE, self.NE:]


    # def _init_neurons(self, rows:int) -> None:
    #     """
    #     @public
    #     Initializes the neurons.
    #     Exc. neurons populate the square of the rows and inh. neurons the square of the halved number of rows.
    #     Assigns arrays to object variables: exc_neurons and inh_neurons, and (joint) neurons
    #     """
    #     self.exc_neurons = np.zeros(int(rows**2))
    #     # self.inh_neurons = np.zeros(int(rows**2 / 4))
    #     # self.neurons = np.append(self.exc_neurons, self.inh_neurons)


    # def populate_grid(self)->np.ndarray:
    #     """
    #     Determines the coordinations of each position in the grid.
    #     Assigns an exc. neuron to each coordination in the grid.
    #     """
    #     N = self.neurons.size
    #     coordinates = np.zeros((N, 2), dtype=int)

    #     exc_coordinates = UNI.get_coordinates(nrows=self.grid.height, step=1)
    #     inh_coordinates = UNI.get_coordinates(nrows=self.grid.height, step=2)
    #     coordinates = np.append(exc_coordinates, inh_coordinates, axis=0)

    #     for neuron in range(len(self.exc_neurons)):
    #         self.grid[coordinates[neuron]] = neuron

    #     return coordinates


    def set_up_neuronal_connections(self, save:bool=True, force:bool=False)->np.ndarray:
        """
        Loads or sets up the connetivity matrix.
        Weighs the synapses.
        """
        cm = ConnectivityMatrix(self._config).load(save=save, force=force)

        W = cm.connections.copy().astype(float)
        W = self._weight_synapses(W)

        return W, cm.connections, cm.shift


    def reset_connectivity_matrix(self)->None:
        self.connectivity_matrix = self._weight_synapses(self.synapses_matrix.copy())


    def _weight_synapses(self, connectivity_matrix:np.ndarray):
        """Weights the synapses according to the exc. and inh. weights, respectively."""
        # NE = self.NE
        connectivity_matrix[:, :self.NE] *= self._synapse.exc_weight
        connectivity_matrix[:, self.NE:] *= self._synapse.inh_weight
        return connectivity_matrix
