# -*- coding: utf-8 -*-
#
# gen_EI_networks_connectivity.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import scipy.io as sio

import lib.connection_matrix as cm
# import lib.protocol as protocol

import matplotlib as mt
from matplotlib.animation import FuncAnimation


import pickle
import matplotlib.pyplot as plt

def save(filename:str, obj:object):
    with open(filename, "wb") as f:
        pickle.dump([obj], f, protocol=-1)



params = {
    "landscape": {
        # "mode": "random",
        # "specs": {
        #     "seed": 0,
        #     "size": 4
        "mode": "independent",
        # "mode": "symmetric",
        # "mode": "Perlin_uniform",
        "specs": {
            "size": 4,
            "base": 1
        }
    },
    "nrowE": 70,
    "nrowI": 35,
    "p": 0.20,
    "seed": 0,
    "shift": 1,
    "stdE": 5,
    "stdI": 5,
    "selfcon": False
}


width = params["nrowE"]
# side = np.arange(width)
snippet = (25, 36)
side = np.arange(*snippet)
X, Y = np.meshgrid(side, side)
coordinates = np.asarray(list(zip(X.ravel(), Y.ravel())))
# coordinates = np.asarray(list(zip(Y.ravel(), X.ravel())))

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


    # # plt.figure("Synapses")
    # post_neurons = np.nonzero(conmat[:, neuron])[0]
    # collections = plt.gca().collections
    # if removal:
    #     while len(collections):
    #         collections.remove(collections[-1])
    # plt.scatter(*coordinates[neuron].T, c=col, s=75)
    # # plt.scatter(*coordinates.T, c="w", s=1)
    # plt.axhline(width - .5)
    # plt.scatter(*coordinates[neuron].T, c="k")
    # plt.scatter(*coordinates[post_neurons].T, c=col)


# def animate_synapses(coordinates:np.ndarray, conmat:np.ndarray):
#     FIG_NAME = "Synapses_animation"
#     fig = plt.figure(FIG_NAME)
#     plot_synapses(coordinates, conmat, 0)
#     def animate(i):
#         plt.figure(FIG_NAME)
#         plot_synapses(coordinates, conmat, i, removal=True)
#         plt.title(f"Neuron: {i}")

#     return FuncAnimation(fig, animate, interval=500, frames=range(width*height-1, 0, -(height+width)//2))




def calculate_direction(x, bins=8, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v

def plot_shift(X=None, Y=None, D=None, name:str=None, **kwargs):
    plt.figure(name, figsize=(4, 3))
    U, V = calculate_direction(D, **kwargs)
    plt.quiver(X, Y, U, V, pivot='middle')



# These are just example configurations. The landscape is taken from the configuration file.
landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random', 'specs': {'seed': 0}},
    {'mode': 'Perlin', 'specs': {'size': 10}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
    {'mode': 'homogeneous', 'specs': {'phi': 6}},
    {'mode': 'independent'},
]

simulation = 'sequence_EI_networks'
# params = protocol.get_parameters(simulation).as_dict()

landscape = params["landscape"]
print(landscape['mode'], params['shift'])
W = cm.EI_networks(**params)

EE, EI, IE, II, shift = W

E_EI = np.concatenate((EE, EI), axis=1)
I_EI = np.concatenate((IE, II), axis=1)
W = np.concatenate((E_EI, I_EI)).T
print(W.shape)


# indegree = IE.T.sum(axis=1).reshape((width, width))
indegree = EE.T.sum(axis=1).reshape((width, width))
indegree_inh = IE.T.sum(axis=1).reshape((width, width))
indegree -= indegree_inh * 4
norm = indegree.min(), indegree.max()
print(norm)
print("max(EE): ", EE.max())
print("max(IE): ", II.max())


save(f"con_matrix_EI_{landscape['mode']}_{landscape['specs']['size']}.bn", (W, shift))

# plt.figure()
# plt.title("Out-degree")
# plt.hist(EE.sum(axis=0))
# plt.figure()
# plt.title("In-degree")
# plt.hist(EE.sum(axis=1))
# plt.figure()
# plt.hist(EE[0], bins=36)
def plot_degree(degree, figname:str=None, title:str=None):
    plt.figure(figname, figsize=(4, 3))
    norm = degree.min(), degree.max()
    plt.imshow(degree, origin="lower", vmin=norm[0], vmax=norm[1], cmap=plt.cm.jet)
    plt.title(title)
    plt.colorbar()
    path = "/home/hauke/"
    plt.savefig(path + figname.replace(".", "-"))




# figname_base = f"{landscape['mode']}_{landscape['specs']['size']}_{landscape['specs']['base']}_{params['stdE']}_{params['stdI']}"
figname_base = f"{landscape['mode']}_{landscape['specs']['size']}_{params['stdE']}_{params['stdI']}"


figname = f"{figname_base}_shift"
sl = slice(*snippet)
shift_r = shift.reshape((width, width))
plot_shift(X, Y, shift_r[sl, sl].flatten(), name=figname)
plt.title(r"Preferred direction $\phi$ of the neurons")
path = "/home/hauke/"
plt.savefig(path + figname.replace(".", "-"))


figname = f"{figname_base}_indegree"
title = "In-degree of the excitatory neurons"
# plt.title(f"In-degree of the excitatory neurons")
plot_degree(indegree, figname=figname, title=title)
# plt.imshow(indegree, origin="lower", vmin=norm[0], vmax=norm[1], cmap=plt.cm.autumn_r)
# plt.colorbar()

# path = "/home/hauke/"
# plt.savefig(path + figname.replace(".", "-"))

figname = f"{figname_base}_outdegree"
# plot_shift(X, Y, shift, name=figname)
outdegree = EE.T.sum(axis=0).reshape((width, width))
# norm = outdegree.min(), outdegree.max()
# plt.imshow(outdegree, origin="lower", vmin=norm[0], vmax=norm[1], cmap=plt.cm.autumn_r)
# plt.colorbar()
title = "Out-degree of the excitatory neurons"
# plt.title(f"Out-degree of the excitatory neurons")
plot_degree(outdegree, figname=figname, title=title)
# plt.savefig(path + figname.replace(".", "-"))

# anim = animate_synapses(coordinates, W)
plot_synapses(EE.T, 2485)
# plot_synapses(coordinates, W, 100)
