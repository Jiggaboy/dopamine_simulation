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

import custom_class.network_configuration as CN
from custom_class.toroid import Toroid
from custom_class.toroid import Coordinate
import gauss_calculator as GAUSS


DEF_VALUE = -1


class Population():

    def __init__(self, NE:int, NI:int, grid_size=tuple):

        if (NE) == grid_size[0] * grid_size[1]:
        # if (NE + NI) == grid_size[0] * grid_size[1]:
            uniform = True
        else:
            uniform = False

        self.exc_neurons = self.set_up_neurons(NE, NeuronType.EXCITATORY)
        self.inh_neurons = self.set_up_neurons(NI, NeuronType.INHIBITORY)
        self.neurons = np.append(self.exc_neurons, self.inh_neurons)

        self.grid = Toroid(grid_size, def_value=DEF_VALUE)
        self.gauss_exc = GAUSS.get_distribution(NeuronType.EXCITATORY)
        self.gauss_inh = GAUSS.get_distribution(NeuronType.INHIBITORY)
        self.coordinates = self.populate_grid(uniform)
        # self.shift = self.get_shift("homogenous", grid_size[0], size=3)

        self.connectivity_matrix = self.set_up_neuronal_connections()


    def __len__(self)->int:
        return len(self.neurons)


    def set_up_neurons(self, amount:int, type_:NeuronType)->np.ndarray:
        return np.full(shape=(amount), fill_value=type_)


    def populate_grid(self, uniform:bool)->np.ndarray:
        N = self.neurons.size
        coordinates = np.zeros((N, 2), dtype=int)

        if uniform:
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
        else:
            for ne in range(N):
                x1, x2 = self.get_empty_slot(DEF_VALUE)
                self.grid[x1, x2] = ne
                coordinates[ne] = Coordinate(x1, x2)
        return coordinates


    def get_empty_slot(self, free_slot_value=None):
        MAX_ITER = 100000
        for _ in range(MAX_ITER):
            x1 = random.randint(0, self.grid.height - 1)
            x2 = random.randint(0, self.grid.width - 1)
            if self.grid[x1, x2] == free_slot_value:
                return x1, x2

    def set_up_neuronal_connections(self, allowSelfConnection:bool=False)->np.ndarray:
        N = self.neurons.size

        W = np.load("W.npy")

        NE = self.exc_neurons.size
        W[:, :NE] *= self.get_synaptic_strength(NeuronType.EXCITATORY)
        W[:, NE:] *= self.get_synaptic_strength(NeuronType.INHIBITORY)
        return W

        W = np.zeros((N, N))
        for pre_neuron in range(N):
            type_ = self.neurons[pre_neuron]
            asymmetric = True
            shift = (0, 0)
            if type_ == NeuronType.EXCITATORY and asymmetric:
                shift = 2 * self.shift[pre_neuron]
            coordinate = self.coordinates[pre_neuron] + shift

            synaptic_strength = self.get_synaptic_strength(self.neurons[pre_neuron])

            for post_neuron in range(N):
                distance = self.grid.get_distance(coordinate, self.coordinates[post_neuron], form="squared")
                connection_probability = self.get_connection_probability(distance, neuron_type=type_)
                if random.random() <= connection_probability:
                    W[post_neuron, pre_neuron] = synaptic_strength

        if not allowSelfConnection:
            for n in range(N):
                W[n, n] = 0
        return W


    def get_connection_probability(self, distance, neuron_type:NeuronType)->float:
        if neuron_type == NeuronType.EXCITATORY:
            return self.gauss_exc[distance]
        else:
            return self.gauss_inh[distance]


    def get_synaptic_strength(self, neuron_type:NeuronType)->float:
        return CF.EXH_STRENGTH if neuron_type == NeuronType.EXCITATORY else CF.INH_STRENGTH


    def get_shift(self, type_:str, nrows:int, *args, **kwargs)->np.ndarray:
        params = {}
        if type_ == "homogenous":
            direction = CN.homogeneous(nrow=nrows, **kwargs)
        elif type_ == "perlin":
            direction = CN.Perlin_uniform(nrow=nrows)
            params["bins"] = 1
        elif type_ == "random":
            direction = CN.random(nrow=nrows)
        shift = CN.shift(direction, **params)
        return shift


    def plot_population(self):
        plt.figure("Neural population")
        col_exc = "red"
        col_inh = "blue"
        # plt.scatter(*self.coordinates[self.neurons == NeuronType.EXCITATORY].T, c=col_exc)
        # plt.scatter(*self.coordinates[self.neurons == NeuronType.INHIBITORY].T, c=col_inh)
        plt.scatter(*self.coordinates.T, c=col_inh)
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


    def save(self, nrows:int):
        fname = CF.POPULATION_FILENAME.format(nrows)
        with open(fname, "wb") as f:
            pickle.dump([self], f, protocol=-1)


    @staticmethod
    def load(nrows:int):
        fname = CF.POPULATION_FILENAME.format(nrows)
        with open(fname, "rb") as f:
            return pickle.load(f)[0]



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
        cls.pop.save(CF.SPACE_WIDTH)
        cls.pop.plot_population()
        cls.pop.hist_in_degree()
        cls.pop.hist_out_degree()
        zeros = cls.pop.connectivity_matrix[cls.pop.connectivity_matrix == 0].size
        nonzeros = cls.pop.connectivity_matrix[cls.pop.connectivity_matrix != 0].size
        exc = cls.pop.connectivity_matrix[cls.pop.connectivity_matrix > 0].size
        inh = cls.pop.connectivity_matrix[cls.pop.connectivity_matrix < 0].size
        print()
        print("==0", zeros)
        print("!=0", nonzeros)
        print(">0", exc)
        print("<0", inh)
        print("total ratio", nonzeros/(nonzeros+zeros))
        N = cls.pop.neurons.size
        # print("exc ratio", exc/(cls.pop.exc_neurons.size * N))
        print("inh ratio", inh/(cls.pop.inh_neurons.size * N))

        print(f"Average activation: {cls.pop.connectivity_matrix.mean()}")

        # CN.plot_shift(*cls.pop.coordinates.T, cls.pop.shift)

    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_no_of_neurons(self):
        self.assertEqual(len(self.pop.exc_neurons), NE)
        self.assertEqual(len(self.pop.inh_neurons), NI)
        self.assertEqual(len(self.pop.neurons), NE+NI)



    # def test_grid_population(self):
    #     self.assertEqual(self.pop.grid.space[self.pop.grid.space != -1].size, NE+NI)
    #     self.assertGreater(self.pop.coordinates[:, 0].sum(), GRID[0])
    #     self.assertGreater(self.pop.coordinates[:, 1].sum(), GRID[1])


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
        self.pop.plot_synapses(100, "y")
        self.pop.plot_synapses(500, "g")
        self.pop.plot_synapses(NE - 1, "r")
        # self.pop.plot_synapses(NE + NI - 1, "r")




if __name__ == '__main__':
    import datetime
    print(datetime.datetime.now())
    unittest.main()
