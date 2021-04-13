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
import dopamine as DOP

from custom_class.neurontype import NeuronType
import custom_class.pickler as PIC
import animation.activity as ACT

from custom_class.toroid import Toroid
from custom_class.toroid import Coordinate


class Population():

    def __init__(self, NE:int, NI:int, grid_size=tuple):
        self.exc_neurons = self.set_up_neurons(NE, NeuronType.EXCITATORY)
        self.inh_neurons = self.set_up_neurons(NI, NeuronType.INHIBITORY)
        self.neurons = np.append(self.exc_neurons, self.inh_neurons)

        self.grid = Toroid(grid_size, def_value=CF.DEF_VALUE)
        self.coordinates = self.populate_grid()

        self.connectivity_matrix, self.synapses_matrix, self.shift = self.set_up_neuronal_connections()
        self.connectivity_cap = self.connectivity_matrix.copy() * CF.CAP


    def __len__(self)->int:
        return len(self.neurons)


    def set_up_neurons(self, amount:int, type_:NeuronType)->np.ndarray:
        return np.full(shape=(amount), fill_value=type_)


    def populate_grid(self)->np.ndarray:
        N = self.neurons.size
        coordinates = np.zeros((N, 2), dtype=int)

        y_grid_positions = np.arange(self.grid.height)
        x_grid_positions = np.arange(self.grid.width)
        x, y = np.meshgrid(x_grid_positions, y_grid_positions)
        exc_coordinates = np.asarray(list(zip(x.ravel(), y.ravel())))
        for neuron in range(NE):
            self.grid[coordinates[neuron]] = neuron

        y_grid_positions = np.arange(0, self.grid.height, 2)
        x_grid_positions = np.arange(0, self.grid.width, 2)
        x, y = np.meshgrid(x_grid_positions, y_grid_positions)
        inh_coordinates = np.asarray(list(zip(x.ravel(), y.ravel())))
        # plt.scatter(*coordinates.T)
        # np.random.shuffle(coordinates)
        coordinates = np.append(exc_coordinates, inh_coordinates, axis=0)
        return coordinates


    def set_up_neuronal_connections(self, allowSelfConnection:bool=False)->np.ndarray:
        W, shift = np.load(f"con_matrix_EI_Perlin_uniform_{CF.PERLIN_SIZE}.bn", allow_pickle=True)[0]
        W = np.asarray(W, dtype=float)
        synapses_matrix = W.copy()

        NE = self.exc_neurons.size
        W[:, :NE] *= self.get_synaptic_strength(NeuronType.EXCITATORY)
        W[:, NE:] *= self.get_synaptic_strength(NeuronType.INHIBITORY)
        return W, synapses_matrix, shift


    def update_synaptic_weights(self, synapses:np.ndarray):
        W = self.connectivity_matrix
        W[synapses, :] = W[synapses, :] * (1 + CF.ETA)
        W[synapses, :] = np.minimum(W[synapses, :], self.connectivity_cap[synapses, :])


    def get_synaptic_strength(self, neuron_type:NeuronType)->float:
        return CF.EXH_STRENGTH if neuron_type == NeuronType.EXCITATORY else CF.INH_STRENGTH


    def plot_population(self):
        plt.figure("Neural population")
        col_exc = "red"
        col_inh = "blue"
        plt.scatter(*self.coordinates[:CF.NE].T, c=col_exc, s=30)
        plt.scatter(*self.coordinates[CF.NE:].T, c=col_inh, s=12)
        msg_template = "{} neurons in {}, in total {};"
        plt.title(msg_template.format("Excitatory", col_exc, self.exc_neurons.size) + "\n"
                  + msg_template.format("Inhibitory", col_inh, self.inh_neurons.size))


    def plot_synapses(self, neuron:int, col:str="r"):
        plt.figure("Synapses")
        post_neurons = np.nonzero(self.connectivity_matrix[:, neuron])[0]
        plt.scatter(*self.coordinates[neuron].T, c=col, s=50)
        plt.scatter(*self.coordinates[neuron].T, c="k", s=20)

        # Do not plot inh. synapses
        post_neurons = post_neurons[post_neurons < self.exc_neurons.size]

        plt.scatter(*self.coordinates[post_neurons].T, c=col, label=neuron)
        plt.legend()


    def plot_shift(self, name:str=None, plot:bool=True, **kwargs):
        U, V = calculate_direction(self.shift, **kwargs)
        if plot:
            plt.figure(name)
            plt.title("Shift of the neurons")
            plt.quiver(*self.coordinates[:CF.NE].T, U, V, pivot='middle')
        # scale = 3
        # U_star = U / scale
        # V_star = V / scale
        # plt.figure()
        # plt.quiver(*self.coordinates[:CF.NE].T, U_star, V, pivot='middle')
        # plt.figure()
        # plt.quiver(*self.coordinates[:CF.NE].T, U, V_star, pivot='middle')
        return U, V


    def polar_shifts(self, patch, name:str=None, **kwargs):
        plt.figure(name)
        directions = 8
        values, _ = np.histogram(self.shift[patch], bins=directions, range=(0, directions))
        # Link last point and first point
        values = np.append(values, values[0])
        c = np.linspace(0, 2*np.pi, directions+1, endpoint=True)
        plt.polar(c, values)
        plt.title(kwargs.get("title"))


    def hist_in_degree(self):
        synapse = self.connectivity_matrix.copy()
        synapse[synapse != 0] = 1
        indegree = synapse.sum(axis=1)
        plt.figure("indegree_histogram")
        plt.hist(indegree)
        plt.title("In-degree of the network")


    def plot_indegree(self):
        indegree = self.synapses_matrix[:CF.NE, :CF.NE].sum(axis=1)
        norm = (indegree.min(), indegree.max())
        figname = "indegree"
        title = "(Excitatory) In-degree of exc. neurons\nOnly synapses are taken into account, not the syn. weights."
        ACT.activity(indegree, norm=norm, figname=figname, title=title)


    def plot_gradient(self):
        synapses_matrix_exc = self.synapses_matrix[:CF.NE, :CF.NE]
        u, v = self.plot_shift(plot=False)
        u_sum = np.zeros(u.shape)
        v_sum = np.zeros(v.shape)
        for n, shift in enumerate(u):
            c = self.coordinates[n]
            patch = DOP.circular_patch(self.grid, center=c, radius=3, coordinates=self.coordinates[:CF.NE])
            u_sum[n] = u[patch].sum()
            v_sum[n] = v[patch].sum()
            if n % 10 == 0:
                print(n)
        u = u_sum.copy()
        v = v_sum.copy()
        jointed = u+v
        PIC.save("u", u)
        PIC.save("v", v)
        PIC.save("uv", jointed)

        # gradient = np.gradient(synapses_matrix_exc)
        # u, v = np.gradient(synapses_matrix_exc)


        # from functools import reduce
        # conv = reduce(np.add,np.gradient(u)) + reduce(np.add,np.gradient(v))

        # gradient = np.add(*gradient)
        norm = (jointed.min(), jointed.max())
        normu = (u.min(), u.max())
        normv = (v.min(), v.max())
        figname = "gradient"
        title = "Gradient of exc. neurons\nOnly synapses are taken into account, not the syn. weights."
        titleu = "Gradient in x-direction\nOnly synapses are taken into account, not the syn. weights."
        titlev = "Gradient in v-direction\nOnly synapses are taken into account, not the syn. weights."
        ACT.activity(u, norm=normu, figname=figname, title=titleu)
        ACT.activity(v, norm=normv, figname="figname", title=titlev)
        ACT.activity(u+v, norm=norm, figname="figname2", title=title)


    def hist_out_degree(self):
        synapse = self.connectivity_matrix.copy()
        synapse[synapse != 0] = 1
        outdegree = synapse.sum(axis=0)
        plt.figure("outdegree_histogram")
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



def calculate_direction(x, bins=8, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v


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


    def test_cap_of_synaptic_updates(self):
        self.assertTrue(hasattr(self.pop, "connectivity_cap"))
        connectivity_diff = self.pop.connectivity_cap[:NE, :NE] - self.pop.connectivity_matrix[:NE, :NE]
        self.assertTrue(np.all(connectivity_diff >= 0))
        synapses = np.arange(10, 30)
        for _ in range(20):
            self.pop.update_synaptic_weights(synapses)
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
    import datetime
    print(datetime.datetime.now())
    unittest.main()
