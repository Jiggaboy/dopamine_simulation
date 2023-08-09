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
import logging
log = logging.getLogger()

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

from connectivitymatrix import ConnectivityMatrix

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    from params import ConnectivityConfig, PerlinConfig, StarterConfig, TestConfig
    config = PerlinConfig()

    conn = create_or_load(config)

    plot_colored_shift(conn.shift)
    plot_shift_arrows(conn.shift)

    ### In- and Outdegrees
    notes = "EE", "EI", "IE", "II"
    mtrx = conn._EE, conn._EI, conn._IE, conn._II

    for n, m in zip(notes, mtrx):
        degrees = conn.degree(m)
        # Normalize
        #for d in degrees:
        #    d /= d.max()
        plot_degree(*degrees, note=n)
        break
    plot_scaled_indegree(conn)

    plt.show()

#===============================================================================
# METHODS
#===============================================================================

def create_or_load(config:object)->object:
    answer = input("Force new connectivity matrix? (y/n)")
    try_load = answer.lower().strip() == "y"
    if try_load:
        log.info(f"Try to load matrix from {config.path_to_connectivity_matrix()}")
    return ConnectivityMatrix(config).load(force=try_load)


def plot_colored_shift(shift):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    plt.figure("SHIFT", figsize=(7, 6))
    im = plt.imshow(shift, origin="lower", cmap=plt.cm.twilight, vmax=8)
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



def plot_degree(*degrees, note:str="undefined", save:bool=False, config:object=None):
    names = "indegree", "outdegree"
    for name, degree in zip(names, degrees):
        info = f"{name}: {note}"
        info = f"{name.capitalize()} of the exc. population"
        fig = plt.figure(info + name + note, figsize=(16, 14))
        plt.title(info)
        im = plt.imshow(degree, origin="lower", cmap=plt.cm.jet)
        plt.colorbar(im, fraction=.046)

        if save:
            plt.savefig(config.sub_dir + f"\{name}.png")


def plot_scaled_indegree(conn_matrix):
    E_indegree, _ = conn_matrix.degree(conn_matrix._EE)
    I_indegree, _ = conn_matrix.degree(conn_matrix._IE)
    indegree = E_indegree - I_indegree * 4

    indegree /= indegree.max()

    plot_degree(indegree, note="scaled")



if __name__ == '__main__':
    main()
