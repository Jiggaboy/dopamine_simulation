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
import lib.universal as UNI

DIRECTIONS = 8
HIST_DEGREE = False

degree_num_kwargs = {
    "figsize": (4.8, 4.), # (2.3, 2.)
    "tight_layout": False,
}




#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    force = UNI.yes_no("Force new connectivity matrix?", False)
    # for shift in [.1, .25, .5, .75, 1., 1.5, 2.0]:
    for shift in [1.]:
        # config.landscape.shift = shift
        # config.landscape.params["base"] = base
        # config.landscape.params["size"] = 2.45
        conn = ConnectivityMatrix(config).load(force=force)

        if shift == 1.:
            plot_colored_shift(conn.shift, note=f"{config.landscape.shift}")
            # plot_shift_arrows(conn.shift)

        import lib.dfs as dfs
        for c in np.arange(8):
            cluster = dfs.find_cluster(conn.shift.reshape((config.rows, config.rows)) == c)
            dim = dfs.get_cluster_dimensions(*cluster)

            print("Mean:", dim.mean(axis=0))
            print("Std:", dim.std(axis=0))
            plt.figure("cluster_dimensions")
            plt.errorbar(*dim.mean(axis=0), *dim.std(axis=0), label=c)
        plt.title("Cluster dimensions")
        plt.xlabel("Width [gridpoints]")
        plt.ylabel("Height [gridpoints]")
        plt.legend()
        plt.show()
        ### In- and Outdegrees
        notes = "EE", "EI", "IE", "II"
        mtrx = conn._EE, conn._EI, conn._IE, conn._II

        for n, m in zip(notes, mtrx):
            # break
            degrees = conn.degree(m)
            degrees = [degree * config.synapse.weight for degree in degrees]

            import lib.dopamine as DOP
            indegree = degrees[0]
            # degree_avg = np.zeros(indegree.shape)

            # for i, row in enumerate(indegree):
            #     for j, elem in enumerate(row):
            #         patch = DOP.circular_patch(config.rows, (i, j), float(6))
            #         patch = patch.reshape((config.rows, config.rows))
            #         degree_avg[i, j] = indegree[patch].mean()

            # plot_degree(degree_avg, note=f"avg_degree_{config.landscape.shift}_{n}", save=True, config=config)
            # for c in config.center_range:
            #     plot_patch(c, config.radius)
            plot_degree(degrees[0], note=f"{config.landscape.shift}_{n}", save=True, config=config)
            for c, center in enumerate(config.center_range.values()):
                plot_patch(center, config.radius[0], width=config.rows)
            # plt.figure()
            # plt.hist(degree_avg.flatten(), bins=25)


            break
        # plot_scaled_indegree(conn, config=config)
    plt.show()


#===============================================================================
# Display Simplex noise conceptually
#===============================================================================
def local_correlation():
    save = True

    config.rows = 20
    config.landscape.params["size"] = 1

    force = UNI.yes_no("Force new connectivity matrix?")
    conn = ConnectivityMatrix(config).load(force=force)

    fig = plt.figure(figsize=(3, 3), num="simplex_noise")
    plot_shift_arrows(conn.shift)
    bins = 6
    plt.xlim(-0.5, bins-0.5)
    plt.xticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plt.ylim(-0.5, bins-0.5)
    plt.yticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plt.tight_layout()
    if save:
        pickler = Pickler(None)
        pickler.save_figure(fig.get_label(), fig, is_general_figure=True)
    plt.show()


#===============================================================================
# METHODS
#===============================================================================

def plot_colored_shift(shift, note:str):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    plt.figure(f"SHIFT_{note}", figsize=(5, 6), tight_layout=True)
    plt.title("shift")
    im = plt.imshow(shift, origin="lower", cmap=plt.cm.hsv, vmax=DIRECTIONS)
    plt.colorbar(im,
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
        fig = plt.figure(info + name + note, **degree_num_kwargs)
        plt.title(info, fontdict={"size": "small"})
        im = plt.imshow(degree, origin="lower", cmap=degree_cmap,
                        # vmin=550, vmax=950,
                        )
        cbar = plt.colorbar(im,
                     # fraction=.04,
                    orientation="vertical",
                    # ticks = [600., 750, 900]
                    )
        # plot_patch(center=(30, 17), radius=6, width=config.rows)
        # plot_patch(center=(36, 38), radius=6, width=config.rows)
        # plt.xticks([0, 30, 60])
        # plt.yticks([0, 30, 60])

        if save:
            pickler = Pickler(config)
            pickler.save_figure(name, fig)


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
