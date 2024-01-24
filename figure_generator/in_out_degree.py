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
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

from lib.connectivitymatrix import ConnectivityMatrix
from plot.lib import plot_patch

DIRECTIONS = 8


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    from params import config
    # for base in np.arange(68, 75):
    for base in [67]:
        # config.landscape.params["base"] = base
        # config.landscape.params["size"] = 3
        conn = create_or_load(config, skip_question=True)

        plot_colored_shift(conn.shift)
        plot_shift_arrows(conn.shift)

        ### In- and Outdegrees
        notes = "EE", "EI", "IE", "II"
        mtrx = conn._EE, conn._EI, conn._IE, conn._II

        for n, m in zip(notes, mtrx):
            break
            degrees = conn.degree(m)
            degrees = [degree  * config.synapse.weight for degree in degrees]
            plot_degree(degrees[0], note=n, save=True, config=config)
            # plot_degree(*degrees, note=n)
            break
        plot_scaled_indegree(conn, config=config)

    plt.show()

#===============================================================================
# METHODS
#===============================================================================

def create_or_load(config:object, skip_question:bool=False)->object:
    if skip_question:
        force = True
    else:
        answer = input("Force new connectivity matrix? (y/n)")
        force = answer.lower().strip() == "y"
        if force:
            logger.info(f"Try to load matrix from {config.path_to_connectivity_matrix()}")
    return ConnectivityMatrix(config).load(force=force)


def plot_colored_shift(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    plt.figure("SHIFT", figsize=(5, 6), tight_layout=True)
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
    plt.quiver(X, Y, U, V, pivot='middle')


def plot_shift_arrows(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    X, Y = np.meshgrid(np.arange(source), np.arange(source))

    plot_shift(X, Y, shift)



def plot_degree(*degrees, note:str="undefined", save:bool=False, config:object=None):
    names = "indegree", "outdegree"
    for name, degree in zip(names, degrees):
        info = f"{name}: {note}"
        info = f"{name.capitalize()} of the exc. population"
        fig = plt.figure(info + name + note, figsize=(5, 6), tight_layout=True)
        plt.title(info)
        im = plt.imshow(degree, origin="lower", cmap=plt.cm.jet)
        plt.colorbar(im,
                     fraction=.04,
                    orientation="horizontal")
        # plot_patch(center=(30, 17), radius=6, width=config.rows)
        # plot_patch(center=(36, 38), radius=6, width=config.rows)

        if save:
            plt.savefig(config.sub_dir + f"\{name}.png")


def plot_scaled_indegree(conn_matrix, config:object):
    E_indegree, _ = conn_matrix.degree(conn_matrix._EE)
    I_indegree, _ = conn_matrix.degree(conn_matrix._IE)
    indegree = E_indegree - I_indegree * 4 #* config.synapse.g
    indegree *= config.synapse.weight

    # indegree /= indegree.max()

    # plot_degree(indegree, note="scaled", config=config)
    plot_degree(indegree, note=f"scaled-{config.landscape.params['base']}", config=config, save=True)



if __name__ == '__main__':
    main()
