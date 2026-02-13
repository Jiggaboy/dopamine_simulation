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
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from params import config

import lib.pickler as PIC
from lib.connectivitymatrix import ConnectivityMatrix
from task import CustomConnectivityMatrix
from plot.lib import plot_patch
import lib.universal as UNI

HIST_DEGREE = False

degree_num_kwargs = {
    "figsize": (5.5, 3.5),
}



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main(config:object):
    force = UNI.yes_no("Force new connectivity matrix?", False)
    conn = ConnectivityMatrix(config, force=force)
    # conn = CustomConnectivityMatrix(config, force=True)
    # conn = CustomConnectivityMatrix(config, force=force)
    
    ## Plot the shift matrix colored by angle
    # fig, ax = plot_colored_shift(conn.shift, note=f"{config.landscape.shift}_{config.landscape.params['base']}", save=False)
    # for name, center in config.center_range.items():
    #     plot_patch(center, config.radius[0], width=config.rows, axis=ax)
    #     ax.text(*center, name, verticalalignment="center", horizontalalignment="center", zorder=12)
    
    
    # fig, ax = plt.subplots(num="Indegree hist")
    # ax.set(title="Indegree across landscape seeds", xlabel="Indegree (# of EE connections)", ylabel="Occurrence")
    # hist_indegrees()
    # return
    
    ### In- and Outdegrees
    notes = "EE", "EI", "IE", "II"
    mtrx = conn._EE, conn._EI, conn._IE, conn._II

    for n, m in zip(notes, mtrx):
        degrees = conn.degree(m)
        degrees = [degree * config.synapse.weight for degree in degrees]
        _, ax= plot_degree(degrees[0], note=f"avg_degree_{config.landscape.shift}_{n}_{config.landscape.params['size']}", save=False, config=config)
        for name, center in config.center_range.items():
            plot_patch(center, config.radius[0], width=config.rows, axis=ax)
            ax.text(*center, name, verticalalignment="center", horizontalalignment="center", zorder=12)
        break

    
    return
    ### DIRECTIONALITY PLOT
    from trial import get_coherence_score
    directionalities = np.zeros((config.rows, config.rows))
    
    for i in np.arange(config.rows):
        for j in np.arange(config.rows):
            # if i == 8 and j == 71:
            shift_matrix = conn.shift.reshape((config.rows, config.rows))
            directionalities[i, j] = get_coherence_score(shift_matrix, (i, j))
    
    plt.figure()
    plt.imshow(-directionalities.T, origin="lower", cmap=plt.cm.Greys, vmin=-.75, vmax=-0.25)
    plt.xlabel("x")
    plt.ylabel("y")
    cbar = plt.colorbar()
    cbar.set_label("-coherence score")
    for name, center in config.center_range.items():
        plot_patch(center, config.radius[0], width=config.rows, axis=plt.gca(), ec="w")
        plt.text(*center, name, verticalalignment="center", horizontalalignment="center", zorder=12, color="w")
    
    

#===============================================================================
# METHODS
#===============================================================================



def hist_indegrees(seed_range:int=5, bins:int=25):
    conns = np.zeros((seed_range, config.no_exc_neurons))
    for s in range(seed_range):
        config.landscape.seed = s
        conn = ConnectivityMatrix(config)
        degrees = conn.degree(conn._EE)
        conns[s] = degrees[0].flatten()
    plt.hist(conns.T, bins=bins)
    
    
def plot_colored_shift(shift, note:str, save:bool=False):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift  = shift.reshape((source, source))
    name = f"SHIFT_{note}"
    fig, ax = plt.subplots(
        num=name,
        # figsize=(4, 5),
        )
    plt.title("Shift")
    plt.xlabel("X-Position")
    plt.ylabel("Y-Position")
    # plt.xticks([10, 40, 70])
    # plt.yticks([10, 40, 70])
    im = plt.imshow(shift.astype(float), origin="lower", cmap=plt.cm.hsv, vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, orientation="vertical")

    if save:
        PIC.save_figure(name, fig, sub_directory=config.sub_dir, transparent=True)
    return fig, ax


def calculate_direction(x, **kwargs):
    u = np.cos(x)
    v = np.sin(x)
    return u, v


def plot_shift(X=None, Y=None, D=None, name:str=None, ax:object=None, **kwargs):
    ax = ax if ax is not None else plt
    # plt.figure(name, figsize=(4, 3))
    U, V = calculate_direction(D, **kwargs)
    ax.quiver(X, Y, U, V, pivot='middle', scale_units="xy", scale=1.5, width=0.01)#units="dots", width=3)


def plot_shift_arrows(shift, **kwargs):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift = shift.reshape((source, source))
    X, Y = np.meshgrid(np.arange(source), np.arange(source))

    plot_shift(X, Y, shift, **kwargs)



def plot_degree(*degrees, note:str="undefined", save:bool=False, config:object=None):
    degree_cmap = plt.cm.jet
    names = "indegree", "outdegree"
    for name, degree in zip(names, degrees):
        info = f"{name.capitalize()} of the exc. population"
        fig, ax = plt.subplots(num=info + name + note, **degree_num_kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # plt.title(info, fontdict={"size": "large"})
        im = plt.imshow(degree,
                        origin="lower",
                        cmap=degree_cmap,
                        # vmin=550, vmax=950,
                        )
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # plt.colorbar(im, cax=cax)

        cbar = plt.colorbar(im,
                    orientation="vertical",
                    # ticks = [600, 750, 900],
                    cax=cax
                    )
        # plot_patch(center=(36, 38), radius=6, width=config.rows)
        cbar.set_label("In-degree", rotation=270, labelpad=15)
        # ax.set_xticks([10, 40, 70])
        # ax.set_yticks([10, 40, 70])
        # plt.tight_layout()

        if save:
            PIC.save_figure(name, fig, sub_directory=config.sub_dir, transparent=True)
        return fig, ax


if __name__ == '__main__':
    # for base in np.arange(100, 125):
    #     config.random_locations = {}
    #     config.landscape.params["base"] = base
    #     # print(base)
    #     main(config)
    main(config)
    plt.show()
