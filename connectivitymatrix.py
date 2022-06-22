# -*- coding: utf-8 -*-
#
# gen_EI_networks_connectivity.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import scipy.io as sio


import logging
log = logging.getLogger()

import lib.connection_matrix as cm
# import lib.protocol as protocol

import matplotlib as mt
from matplotlib.animation import FuncAnimation

from time import perf_counter
from util import pickler as PIC
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 20

# Possible values for the landscape are:
#     - random (Random preferred direction, seed:int)
#     - independent (Does not have a preferred direction nor a spatial distribution, seed:int)
#     - symmetric (No pref. direction, seed:int)
#     - Perlin (Pref. direction according to the Perlin noise) {Parameter: size:int, base:int, seed:int}
#     - Perlin_uniform (as for Perlin but euqally distributed throughout the 8 directions)
#     - homogeneous (all have the same preferred direction) {Takes a preferred direction 'phi' with values between 0-7, seed:int}



class ConnectivityMatrix:

    @property
    def connections(self):
        E_EI = np.concatenate((self._EE, self._EI), axis=1)
        I_EI = np.concatenate((self._IE, self._II), axis=1)
        W = np.concatenate((E_EI, I_EI)).T
        return W


    def __init__(self, config):
        log.info("Initialize ConnectivityMatrix…")
        self._rows = config.rows
        self._landscape = config.landscape
        self._path = config.path_to_connectivity_matrix()


    def connect_neurons(self, save:bool=True):
        log.info("Connect Neurons…")
        self._EE, self._EI, self._IE, self._II, self.shift = cm.EI_networks(self._landscape, self._rows)

        log.info("Check for self connections")
        assert np.all(np.diagonal(self._EE) == 0)
        assert np.all(np.diagonal(self._II) == 0)

        if save:
            log.info(f"Save connectivity matrix to: {self._path}")
            PIC.save(self._path, self)


    @classmethod
    def load(cls, config):
        path = config.path_to_connectivity_matrix()
        log.info(f"Load connectivity matrix from {path}…")
        return PIC.load(path)


    @staticmethod
    def degree(matrix:np.ndarray):
        source, target = np.sqrt(matrix.shape).astype(int)
        indegree = matrix.sum(axis=0).reshape((target, target))
        outdegree = matrix.sum(axis=1).reshape((source, source))
        return indegree, outdegree



def plot_degree(*degrees, note:str="undefined"):
    names = "indegree", "outdegree"
    for name, degree in zip(names, degrees):
        info = f"{name}: {note}"
        info = f"{name.capitalize()} of the exc. population"
        plt.figure(info + name + note, figsize=(7, 6))
        plt.title(info)
        im = plt.imshow(degree, origin="lower", cmap=plt.cm.jet)
        plt.colorbar(im, fraction=.046)

        from figure_generator.connectivity_distribution import set_layout
        set_layout(margin=0)


def plot_scaled_indegree(conn_matrix):
    E_indegree, _ = conn_matrix.degree(conn_matrix._EE)
    I_indegree, _ = conn_matrix.degree(conn_matrix._IE)
    indegree = E_indegree - I_indegree * 4

    indegree /= indegree.max()

    plot_degree(indegree, note="scaled")




def plot_colored_shift(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    plt.figure("SHIFT", figsize=(7, 6))
    im = plt.imshow(shift, origin="lower", cmap=plt.cm.hsv, vmax=8)
    plt.colorbar(im, fraction=.046)


def calculate_direction(x, bins=8, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v

def plot_shift(X=None, Y=None, D=None, name:str=None, **kwargs):
    # plt.figure(name, figsize=(4, 3))
    U, V = calculate_direction(D, **kwargs)
    plt.quiver(X, Y, U, V, pivot='middle')

def plot_shift_arrows(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    X, Y = np.meshgrid(np.arange(source), np.arange(source))

    plot_shift(X, Y, shift)

before = perf_counter()


if __name__ == "__main__":
    print("Run as main…")
    from params import ConnectivityConfig, PerlinConfig, StarterConfig

    Config = ConnectivityConfig()
    Config = PerlinConfig()
    print(f"Weight: {Config.synapse.weight} and prob. {Config.landscape.connection_probability}")

    ## Either create a new one or load it
    try_load = input("Load connectivity matrix? (y/n)")
    if try_load.lower().strip() == "y":
        conn = ConnectivityMatrix.load(Config)
    else:
        conn = ConnectivityMatrix(Config)
        conn.connect_neurons()

    # Shift
    plot_colored_shift(conn.shift)
    plot_shift_arrows(conn.shift)

    # conn._EE[-1500:, :] = .01

    ### In- and Outdegrees
    notes = "EE", "EI", "IE", "II"
    mtrx = conn._EE, conn._EI, conn._IE, conn._II
    for n, m in zip(notes, mtrx):
        degrees = conn.degree(m)
        # Normalize
        # for d in degrees:
        #     d /= d.max()
        plot_degree(*degrees, note=n)

        def sqr(m):
            return m.reshape([70, 70])

        indegree = m.sum(axis=0)
        plot_degree(sqr(indegree), note="plain - 1st")

        scaled = indegree
        for i in range(3):
            scaled = m.T.dot(scaled)
            # scaled = scaled.dot(m.T)
            plot_degree(sqr(scaled), note=f"iter: {i + 2}")

        break # Only plot EE

    # plot_scaled_indegree(conn)
after = perf_counter()

print(f"Time elapsed: {after - before}")
plt.show()

############### To be updated


# side = np.arange(70)
# snippet = (25, 36)
# side = np.arange(*snippet)
# X, Y = np.meshgrid(side, side)
# coordinates = np.asarray(list(zip(X.ravel(), Y.ravel())))
# # coordinates = np.asarray(list(zip(Y.ravel(), X.ravel())))







# figname = f"_shift"
# sl = slice(*snippet)
# shift_r = shift.reshape((width, width))
# plot_shift(X, Y, shift_r[sl, sl].flatten(), name=figname)
# plt.title(r"Preferred direction $\phi$ of the neurons")
# path = "/home/hauke/"
# plt.savefig(path + figname.replace(".", "-"))


# plot_synapses(EE.T, 2485)

    # plt.figure("synapses", figsize=(4, 4))
    # colormap = plt.cm.Blues
    # indegree_of_single_neuron = conn._EE[:, 100]
    # indegree_of_single_neuron = conn._EE[100]
    # # indegree_of_single_neuron = conn._II[:, 1]
    # # indegree_of_single_neuron = conn._II[100]
    # width = int(np.sqrt(indegree_of_single_neuron.size))
    # norm = indegree_of_single_neuron.min(), indegree_of_single_neuron.max()
    # plt.imshow(indegree_of_single_neuron.reshape((width, width)), origin="lower", cmap=colormap, vmin=norm[0], vmax=norm[1])
    # cbar_props = plt.cm.ScalarMappable(norm=mt.colors.Normalize(*norm), cmap=colormap)
    # plt.colorbar(cbar_props)





def plot_synapses(conmat, neuron:int, col:str="r", removal:bool=False):
    plt.figure("synapses", figsize=(4, 3))
    colormap = plt.cm.Blues

    degree = conmat[:, neuron]
    norm = degree.min(), degree.max()
    plt.imshow(degree.reshape(width, width), origin="lower", cmap=colormap, vmin=norm[0], vmax=norm[1])
    cbar_props = plt.cm.ScalarMappable(norm=mt.colors.Normalize(*norm), cmap=colormap)
    plt.colorbar(cbar_props)
    plt.title(f"Axonal connections of neuron {neuron}")
    # degree[neuron] = degree.max() * 2
    # image.set_data(degree.reshape(width, width))


#     # # plt.figure("Synapses")
#     # post_neurons = np.nonzero(conmat[:, neuron])[0]
#     # collections = plt.gca().collections
#     # if removal:
#     #     while len(collections):
#     #         collections.remove(collections[-1])
#     # plt.scatter(*coordinates[neuron].T, c=col, s=75)
#     # # plt.scatter(*coordinates.T, c="w", s=1)
#     # plt.axhline(width - .5)
#     # plt.scatter(*coordinates[neuron].T, c="k")
#     # plt.scatter(*coordinates[post_neurons].T, c=col)


# # def animate_synapses(coordinates:np.ndarray, conmat:np.ndarray):
# #     FIG_NAME = "Synapses_animation"
# #     fig = plt.figure(FIG_NAME)
# #     plot_synapses(coordinates, conmat, 0)
# #     def animate(i):
# #         plt.figure(FIG_NAME)
# #         plot_synapses(coordinates, conmat, i, removal=True)
# #         plt.title(f"Neuron: {i}")

# #     return FuncAnimation(fig, animate, interval=500, frames=range(width*height-1, 0, -(height+width)//2))
