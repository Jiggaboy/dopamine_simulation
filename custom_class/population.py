#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:24:19 2021

@author: hauke
"""



import cflogger
log = cflogger.getLogger()


import numpy as np

from custom_class.neurontype import NeuronType
from custom_class.toroid import Toroid
from connectivitymatrix import ConnectivityMatrix
import lib.lib.dopamine as DOP
from lib import SingletonClass
import lib.universal as UNI


class Population(SingletonClass):

    def __init__(self, config):
        super().__init__()
        log.info("Create new Population…")
        self._config = config
        self._landscape = config.landscape
        self._synapse = config.synapse

        self._init_neurons(self._config.rows)

        self.grid = Toroid(self._config.rows)
        self.coordinates = self.populate_grid()

        self.connectivity_matrix, self.synapses_matrix, self.shift = self.set_up_neuronal_connections(landscape=self._landscape)


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


    def set_up_neuronal_connections(self, landscape)->np.ndarray:
        """
        Loads or sets up the connetivity matrix.
        Weighs the synapses.
        """
        cm = ConnectivityMatrix(self._config).load(save=True)

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


    ################# MOVED TO UNIVERSAL ####################################################################
    # @staticmethod
    # def populate_subgrid(height:int, width:int, step:int=1)->np.ndarray:
    #     """Returns the coordinates of the grid in the steplength {step}."""
    #     y_grid_positions = np.arange(0, height, step)
    #     x_grid_positions = np.arange(0, width, step)
    #     x, y = np.meshgrid(x_grid_positions, y_grid_positions)
    #     coordinates = np.asarray(list(zip(x.ravel(), y.ravel())))
    #     return coordinates
    ################# END ####################################################################################

    # def plot_population(self):
    #     plt.figure("Neural population")
    #     col_exc = "red"
    #     col_inh = "blue"
    #     plt.scatter(*self.coordinates[:CF.NE].T, c=col_exc, s=30)
    #     plt.scatter(*self.coordinates[CF.NE:].T, c=col_inh, s=12)
    #     msg_template = "{} neurons in {}, in total {};"
    #     plt.title(msg_template.format("Excitatory", col_exc, self.exc_neurons.size) + "\n"
    #               + msg_template.format("Inhibitory", col_inh, self.inh_neurons.size))

    # def plot_synapses(self, neuron: int, col: str = "r"):
    #     plt.figure("Synapses")
    #     post_neurons = np.nonzero(self.connectivity_matrix[:, neuron])[0]
    #     plt.scatter(*self.coordinates[neuron].T, c=col, s=50)
    #     plt.scatter(*self.coordinates[neuron].T, c="k", s=20)

    #     # Do not plot inh. synapses
    #     post_neurons = post_neurons[post_neurons < self.exc_neurons.size]

    #     plt.scatter(*self.coordinates[post_neurons].T, c=col, label=neuron)
    #     plt.legend()

    # def plot_shift(self, name: str = None, plot: bool = True, **kwargs):
    #     U, V = calculate_direction(self.shift, **kwargs)
    #     if plot:
    #         plt.figure(name)
    #         plt.title("Shift of the neurons")
    #         plt.quiver(*self.coordinates[:CF.NE].T, U, V, pivot='middle')
    #     scale = 3
    #     U_star = U / scale
    #     V_star = V / scale
    #     plt.figure()
    #     plt.quiver(*self.coordinates[:CF.NE].T, U_star, V, pivot='middle')
    #     plt.figure()
    #     plt.quiver(*self.coordinates[:CF.NE].T, U, V_star, pivot='middle')
    #     return U, V


    # def polar_shifts(self, patch, name:str=None, **kwargs):
    #     plt.figure(name)
    #     directions = 8
    #     values, _ = np.histogram(self.shift[patch], bins=directions, range=(0, directions))
    #     # Link last point and first point
    #     values = np.append(values, values[0])
    #     c = np.linspace(0, 2*np.pi, directions+1, endpoint=True)
    #     plt.polar(c, values)
    #     plt.title(kwargs.get("title"))


    # def hist_in_degree(self):
    #     synapse = self.connectivity_matrix.copy()
    #     synapse[synapse != 0] = 1
    #     indegree = synapse.sum(axis=1)
    #     plt.figure("indegree_histogram")
    #     plt.hist(indegree)
    #     plt.title("In-degree of the network")


### TEST
import unittest


class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pop = Population(StarterConfig())


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

    def test_no_of_neurons(self):
        self.assertEqual(len(self.pop.exc_neurons), NE)
        self.assertEqual(len(self.pop.inh_neurons), NI)
        self.assertEqual(len(self.pop.neurons), NE+NI)


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


    def test_cap_of_synaptic_updates(self):
        self.assertTrue(hasattr(self.pop, "connectivity_cap"))
        connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
        self.assertTrue(np.all(connectivity_diff >= 0))
        synapses = np.arange(10, 30)
        for _ in range(20):
            self.pop.update_synaptic_weights(synapses, learning_rate=.01)
        connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
        self.assertTrue(np.all(connectivity_diff >= 0))


    def test_shift(self):
        self.assertTrue(hasattr(self.pop, "shift"), "No shift attribute...")
        self.assertEqual(self.pop.shift.size, self.pop.exc_neurons.size)
        self.pop.plot_shift()

        n = 100
        n_cross = DOP.circular_patch(CF.SPACE_WIDTH, self.pop.coordinates[n], radius=5)
        self.pop.polar_shifts(n_cross, title=f"Neuron {n}")


    def test_plot_indegree(self):
        self.assertTrue(hasattr(self.pop, "plot_indegree"), "No plot_indegree attribute...")
        self.pop.plot_indegree()


    def _test_plot_gradient(self):
        self.assertTrue(hasattr(self.pop, "plot_gradient"), "No plot_gradient attribute...")
        self.pop.plot_gradient()




if __name__ == '__main__':
    from params import StarterConfig
    unittest.main()
