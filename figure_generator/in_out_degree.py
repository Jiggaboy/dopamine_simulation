#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
from params import config

from lib.pickler_class import Pickler
from lib.connectivitymatrix import ConnectivityMatrix
from plot.lib import plot_patch

DIRECTIONS = 8
HIST_DEGREE = False


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    for _ in [0]:
    # for base in np.arange(18, 24):
        # config.landscape.params["base"] = base
        # config.landscape.params["size"] = 2.45
        conn = create_or_load(config, force=None)

        # plot_colored_shift(conn.shift, note=f"{config.landscape.params['base']}-{config.landscape.params['size']}")
        # plot_shift_arrows(conn.shift)

        ### In- and Outdegrees
        notes = "EE", "EI", "IE", "II"
        mtrx = conn._EE, conn._EI, conn._IE, conn._II

        for n, m in zip(notes, mtrx):
            # break
            degrees = conn.degree(m)
            degrees = [degree  * config.synapse.weight for degree in degrees]
            plot_degree(degrees[0], note=n, save=True, config=config)
            # plot_degree(*degrees, note=n, config=config)
            break
        # plot_scaled_indegree(conn, config=config)

    plt.show()


# def main():
#     save = True

#     config.rows = 20
#     config.landscape.params["size"] = 1
#     conn = create_or_load(config, force=None)
#     fig = plt.figure(figsize=(3, 3), num="simplex_noise")
#     plot_shift_arrows(conn.shift)
#     bins = 6
#     plt.xlim(-0.5, bins-0.5)
#     plt.xticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
#     plt.ylim(-0.5, bins-0.5)
#     plt.yticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
#     plt.tight_layout()
#     if save:
#         pickler = Pickler(None)
#         pickler.save_figure(fig.get_label(), fig, is_general_figure=True)
#     plt.show()


#===============================================================================
# METHODS
#===============================================================================

def create_or_load(config:object, force:bool=None)->object:
    if force == None:
        answer = input("Force new connectivity matrix? (y/n)")
        force = answer.lower().strip() == "y"
    if force:
        logger.info(f"Try to load matrix from {config.path_to_connectivity_matrix()}")
    return ConnectivityMatrix(config).load(force=force)


def plot_colored_shift(shift, note:str):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    plt.figure(f"SHIFT_{note}", figsize=(5, 6), tight_layout=True)
    plt.title("shift")
    im = plt.imshow(shift, origin="lower", cmap=plt.cm.twilight, vmax=DIRECTIONS)
    plt.colorbar(im,
                  fraction=.04,
                 orientation="horizontal")


def calculate_direction(x, bins=DIRECTIONS, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v


def plot_shift(X=None, Y=None, D=None, name:str=None, **kwargs):
    # plt.figure(name, figsize=(4, 3))
    U, V = calculate_direction(D, **kwargs)
    plt.quiver(X, Y, U, V, pivot='middle', scale_units="xy", scale=1.33, units="dots", width=3)


def plot_shift_arrows(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    X, Y = np.meshgrid(np.arange(source), np.arange(source))

    plot_shift(X, Y, shift)



def plot_degree(*degrees, note:str="undefined", save:bool=False, config:object=None):
    degree_cmap = plt.cm.jet
    names = "indegree", "outdegree"
    for name, degree in zip(names, degrees):
        info = f"{name.capitalize()} of the \nexc. population"
        fig = plt.figure(info + name + note, figsize=(2.3, 2.), tight_layout=False)
        plt.title(info, fontdict={"size": "small"})
        im = plt.imshow(degree, origin="lower", cmap=degree_cmap, )
        cbar = plt.colorbar(im,
                     # fraction=.04,
                    orientation="vertical",
                    ticks = [600., 750, 900])
        # plot_patch(center=(30, 17), radius=6, width=config.rows)
        # plot_patch(center=(36, 38), radius=6, width=config.rows)
        plt.xticks([0, 30, 60])
        plt.yticks([0, 30, 60])


        if save:
            pickler = Pickler(config)
            pickler.save_figure(name, fig)
            # plt.savefig(config.sub_dir + f"\{name}.png")


def plot_scaled_indegree(conn_matrix, config:object):
    E_indegree, _ = conn_matrix.degree(conn_matrix._EE)
    I_indegree, _ = conn_matrix.degree(conn_matrix._IE)
    indegree = E_indegree - I_indegree * 4 #* config.synapse.g
    indegree *= config.synapse.weight

    # indegree /= indegree.max()

    note = f"scaled-{config.landscape.params['base']}-{config.landscape.params['size']}"
    plot_degree(indegree, note=note, config=config, save=True)
    if HIST_DEGREE:
        hist_degree(indegree, note=note)


def hist_degree(degree:np.ndarray, bins:np.ndarray=None, note:str=None) -> None:
    bins = np.linspace(-220, 250) if bins is None else bins
    plt.figure("hist_" + note)
    plt.hist(degree.ravel(), bins=bins)
    plt.ylim(0, 350)


if __name__ == '__main__':
    main()
