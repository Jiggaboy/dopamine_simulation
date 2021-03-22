#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:24:19 2021

@author: hauke
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

import configuration as CF

from custom_class.neurontype import NeuronType
import custom_class.pickler as PIC

from custom_class.toroid import Toroid
from custom_class.toroid import Coordinate


class Population():

    def __init__(self, NE:int, NI:int, grid_size=tuple):
        self.exc_neurons = self.set_up_neurons(NE, NeuronType.EXCITATORY)
        self.inh_neurons = self.set_up_neurons(NI, NeuronType.INHIBITORY)
        self.neurons = np.append(self.exc_neurons, self.inh_neurons)

        self.grid = Toroid(grid_size, def_value=CF.DEF_VALUE)
        self.coordinates = self.populate_grid()

        self.connectivity_matrix = self.set_up_neuronal_connections()


    def __len__(self)->int:
        return len(self.neurons)


    def set_up_neurons(self, amount:int, type_:NeuronType)->np.ndarray:
        return np.full(shape=(amount), fill_value=type_)


    def populate_grid(self)->np.ndarray:
        N = self.neurons.size
        coordinates = np.zeros((N, 2), dtype=int)

        y_grid_positions = np.arange(self.grid.height)
        x_grid_positions = np.arange(self.grid.width)
        # x_grid_positions = np.arange(self.grid.height)
        # y_grid_positions = np.arange(self.grid.width)
        x, y = np.meshgrid(x_grid_positions, y_grid_positions)
        # coordinates = np.asarray(list(zip(y.ravel(), x.ravel())))
        coordinates = np.asarray(list(zip(x.ravel(), y.ravel())))
        # plt.scatter(*coordinates.T)
        # np.random.shuffle(coordinates)
        for neuron in range(NE):
            self.grid[coordinates[neuron][0], coordinates[neuron][1]] = neuron
        return coordinates

    def set_up_neuronal_connections(self, allowSelfConnection:bool=False)->np.ndarray:
        W = np.load("con_matrix_EI_Perlin_uniform.bn", allow_pickle=True)[0]

        NE = self.exc_neurons.size
        W[:, :NE] *= self.get_synaptic_strength(NeuronType.EXCITATORY)
        W[:, NE:] *= self.get_synaptic_strength(NeuronType.INHIBITORY)
        return W


    def get_synaptic_strength(self, neuron_type:NeuronType)->float:
        return CF.EXH_STRENGTH if neuron_type == NeuronType.EXCITATORY else CF.INH_STRENGTH


    def plot_population(self):
        plt.figure("Neural population")
        col_exc = "red"
        col_inh = "blue"
        plt.scatter(*self.coordinates.T, c=col_exc)
        msg_template = "{} neurons in {}, in total {};"
        plt.title(msg_template.format("Excitatory", col_exc, self.exc_neurons.size) + "\n"
                  + msg_template.format("Inhibitory", col_inh, self.inh_neurons.size))


    def plot_synapses(self, neuron:int, col:str="r"):
        plt.figure("Synapses")
        post_neurons = np.nonzero(self.connectivity_matrix[:, neuron])[0]
        plt.scatter(*self.coordinates[neuron].T, c=col, s=75)
        plt.scatter(*self.coordinates[neuron].T, c="k", s=35)

        # Do not plot inh. synapses
        post_neurons = post_neurons[post_neurons < self.exc_neurons.size]

        plt.scatter(*self.coordinates[post_neurons].T, c=col)


    def hist_in_degree(self):
        synapse = self.connectivity_matrix.copy()
        synapse[synapse != 0] = 1
        indegree = synapse.sum(axis=1)
        plt.figure("indegree")
        plt.hist(indegree)
        plt.title("In-degree of the network")


    def hist_out_degree(self):
        synapse = self.connectivity_matrix.copy()
        synapse[synapse != 0] = 1
        outdegree = synapse.sum(axis=0)
        plt.figure("outdegree")
        plt.hist(outdegree)
        plt.title("Out-degree of the network")


    def save(self, nrows:int, terminated:bool=False):
        pop_id = assemble_population_id(nrows, terminated)
        fname = CF.POPULATION_FILENAME.format(pop_id)
        PIC.save(fname, self)


    @staticmethod
    def load(nrows:int, terminated:bool=False):
        pop_id = assemble_population_id(nrows, terminated)
        fname = CF.POPULATION_FILENAME.format(pop_id)
        return PIC.load(fname)


def assemble_population_id(nrows:int, terminated:bool=False)->str:
    if not terminated:
        pop_id = nrows
    else:
        pop_id = str(nrows) + "_final"
    return pop_id




### TEST
import unittest

NE = CF.NE
NI = CF.NI
GRID = (CF.SPACE_WIDTH, CF.SPACE_HEIGHT)


class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pop = Population(NE, NI, GRID)

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


    def test_plot_synapses(self):
        self.pop.plot_synapses(0, "y")
        self.pop.plot_synapses(1200, "g")
        self.pop.plot_synapses(NE - 1, "r")


    def test_save_and_load(self):
        self.pop.save(CF.SPACE_WIDTH)
        self.pop.save(CF.SPACE_WIDTH, terminated=True)
        Population.load(CF.SPACE_WIDTH)
        Population.load(CF.SPACE_WIDTH, terminated=True)





if __name__ == '__main__':
    import datetime
    print(datetime.datetime.now())
    unittest.main()
