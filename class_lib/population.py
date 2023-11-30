#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:24:19 2021

@author: hauke
"""



import cflogger
log = cflogger.getLogger()


import numpy as np

from class_lib.neurontype import NeuronType
from class_lib.toroid import Toroid
from lib.connectivitymatrix import ConnectivityMatrix
import lib.dopamine as DOP
from lib import SingletonClass
import lib.universal as UNI


class Population(SingletonClass):

    def __init__(self, config, save:bool=True, force:bool=False):
        super().__init__()
        log.info("Create new Population…")
        self._config = config
        self._landscape = config.landscape
        self._synapse = config.synapse

        self._init_neurons(self._config.rows)

        self.grid = Toroid(self._config.rows)
        self.coordinates = self.populate_grid()

        self.connectivity_matrix, self.synapses_matrix, self.shift = self.set_up_neuronal_connections(save, force)


    def __len__(self)->int:
        return len(self.neurons)


    @property
    def EE_connections(self):
        return self.connectivity_matrix[:self.exc_neurons.size, :self.exc_neurons.size]


    @property
    def II_connections(self):
        return self.connectivity_matrix[self.exc_neurons.size:, self.exc_neurons.size:]


    @property
    def IE_connections(self):
        # Target: source notation
        return self.connectivity_matrix[self.exc_neurons.size:, :self.exc_neurons.size]


    @property
    def EI_connections(self):
        return self.connectivity_matrix[:self.exc_neurons.size, self.exc_neurons.size:]


    def _init_neurons(self, rows:int):
        """
        Initializes the neurons.
        Exc. neurons populate the square of the rows and inh. neurons the square of the halved number of rows.
        Assigns arrays to object variables: exc_neurons and inh_neurons, and (joint) neurons
        """
        NE = rows**2
        self.exc_neurons = self._set_up_neurons(NE, NeuronType.EXCITATORY)
        NI = (rows // 2)**2
        self.inh_neurons = self._set_up_neurons(NI, NeuronType.INHIBITORY)
        self.neurons = np.append(self.exc_neurons, self.inh_neurons)


    def populate_grid(self)->np.ndarray:
        """
        Determines the coordinations of each position in the grid.
        Assigns an exc. neuron to each coordination in the grid.
        """
        N = self.neurons.size
        coordinates = np.zeros((N, 2), dtype=int)

        exc_coordinates = UNI.get_coordinates(nrows=self.grid.height, step=1)
        inh_coordinates = UNI.get_coordinates(nrows=self.grid.height, step=2)
        coordinates = np.append(exc_coordinates, inh_coordinates, axis=0)

        for neuron in range(len(self.exc_neurons)):
            self.grid[coordinates[neuron]] = neuron

        return coordinates


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
        NE = self.exc_neurons.size
        connectivity_matrix[:, :NE] *= self._synapse.exc_weight
        connectivity_matrix[:, NE:] *= self._synapse.inh_weight
        return connectivity_matrix


    def update_synaptic_weights(self, synapses:np.ndarray, learning_rate:float):
        W = self.connectivity_matrix
        W[synapses, :] = W[synapses, :] * (1 + learning_rate)
        W[synapses, :] = np.minimum(W[synapses, :], self.connectivity_cap[synapses, :])



    @staticmethod
    def _set_up_neurons(amount:int, type_:NeuronType)->np.ndarray:
        return np.full(shape=(amount), fill_value=type_)



### TEST
import unittest


class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pop = Population(TestConfig())


    @classmethod
    def tearDownClass(cls):
        cls.pop.plot_population()
        cls.pop.hist_in_degree()
        cls.pop.hist_out_degree()
        print()
        print(f"Average activation: {cls.pop.connectivity_matrix.mean()}")


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_no_self_connection(self):
        for idx, n in enumerate(self.pop.connectivity_matrix):
            self.assertFalse(n[idx], msg=f"{idx} has a self connection!")


    def test_connectivity(self):
        self.assertGreater(self.pop.connectivity_matrix.nonzero()[0].size, self.pop.neurons.size)

        for axon in self.pop.connectivity_matrix.T:
            synapses = axon[axon != 0]
            excitatory = all(synapses > 0)
            inhibitory = all(synapses < 0)
            self.assertTrue(excitatory or inhibitory)


    # def test_cap_of_synaptic_updates(self):
    #     self.assertTrue(hasattr(self.pop, "connectivity_cap"))
    #     connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
    #     self.assertTrue(np.all(connectivity_diff >= 0))
    #     synapses = np.arange(10, 30)
    #     for _ in range(20):
    #         self.pop.update_synaptic_weights(synapses, learning_rate=.01)
    #     connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
    #     self.assertTrue(np.all(connectivity_diff >= 0))


    def test_plot_indegree(self):
        self.assertTrue(hasattr(self.pop, "plot_indegree"), "No plot_indegree attribute...")
        self.pop.plot_indegree()


    def _test_plot_gradient(self):
        self.assertTrue(hasattr(self.pop, "plot_gradient"), "No plot_gradient attribute...")
        self.pop.plot_gradient()




if __name__ == '__main__':
    from params import TestConfig
    unittest.main()
