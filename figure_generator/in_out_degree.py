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
from matplotlib import rcParams
from params import config

import lib.pickler as PIC
from lib.connectivitymatrix import ConnectivityMatrix
from plot.lib import plot_patch
import lib.universal as UNI

DIRECTIONS = 8
HIST_DEGREE = False

degree_num_kwargs = {
    "figsize": (4.5, 3.5),
    "tight_layout": True,
}

rcParams["font.size"] = 12
rcParams["figure.figsize"] = (3.5, 3.5)


def asymmetry_score(A:np.ndarray) -> float:
    return 0.5 * np.linalg.norm(A - A.T) / np.linalg.norm(A)

def asymmetry_score(A:np.ndarray) -> float:
    sym = np.linalg.norm(0.5 * (A + A.T))
    asym= np.linalg.norm(0.5 * (A - A.T))
    return (sym - asym) / (sym + asym)

eye = np.eye(10)
print(asymmetry_score(eye))
print(asymmetry_score(1-eye))

# print(np.roll(eye, shift=1, axis=0))
print(asymmetry_score(np.roll(eye, shift=1, axis=0)))
# quit()

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    # seed_range = 5
    # conns = np.zeros((seed_range, config.no_exc_neurons))
    # for s in range(seed_range):
    #     config.landscape.seed = s
    #     conn = ConnectivityMatrix(config).load()
    #     degrees = conn.degree(conn._EE)
    #     conns[s] = degrees[0].flatten()
    # plt.figure("Indegree hist")
    # plt.title("Indegree across landscape seeds")
    # plt.xlabel("Indegree (# of EE connections)")
    # plt.ylabel("occurrence")
    # plt.hist(conns.T, bins=25)
    # plt.show()
    # return


    force = UNI.yes_no("Force new connectivity matrix?", False)

    shifts = np.asarray([0.1, 1., 2.])
    sizes = np.asarray([0.5, 1, 2, 5, 10, 25])

    score = np.zeros((shifts.size, sizes.size))
    score_random = np.zeros((shifts.size, sizes.size))
    config.landscape.params["base"] = 4
    for s, shift in enumerate(shifts):
        for l, size in enumerate(sizes):
            config.landscape.shift = shift
            config.landscape.params["size"] = size
            conn = ConnectivityMatrix(config).load(force=force)

            if shift == 1.:
                plot_colored_shift(conn.shift, note=f"{config.landscape.shift}_{config.landscape.params['size']}", save=False)
                # plot_shift_arrows(conn.shift)
                # local_correlation()

            # import lib.dfs as dfs
            # name = "cluster_dimensions"
            # fig = plt.figure(name)
            # for c in np.arange(8):
            #     cluster = dfs.find_cluster(conn.shift.reshape((config.rows, config.rows)) == c)
            #     dim = dfs.get_cluster_dimensions(*cluster)

            #     print("Mean:", dim.mean(axis=0))
            #     print("Std:", dim.std(axis=0))
            #     plt.errorbar(*dim.mean(axis=0), *dim.std(axis=0), label=c)
            # plt.title("Cluster dimensions")
            # plt.xlabel("Width [gridpoints]")
            # plt.ylabel("Height [gridpoints]")
            # plt.legend()
            # PIC.save_figure(name, fig, sub_directory=config.sub_dir)
            # plt.show()

            ### In- and Outdegrees
            notes = "EE", "EI", "IE", "II"
            mtrx = conn._EE, conn._EI, conn._IE, conn._II

            for n, m in zip(notes, mtrx):
                # break
                degrees = conn.degree(m)
                degrees = [degree * config.synapse.weight for degree in degrees]
                # plot_degree(degrees[0], note=f"avg_degree_{config.landscape.shift}_{n}_{config.landscape.params['size']}", save=True, config=config)

                print("ASYMMETRY SCORE")
                print(size, shift, asymmetry_score(m))
                score[s, l] = asymmetry_score(m)
                break
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



                break
            # plot_scaled_indegree(conn, config=config)

    config.landscape.shift = 0.
    conn = ConnectivityMatrix(config).load(force=force)
    symm = asymmetry_score(conn._EE)

    plt.figure("Asymmetry score")
    plt.xlabel("Perlin size")
    plt.ylabel("Asymmetry score")
    plt.axhline(symm, c="k", label="Symmetric/No shift")
    for i, sc in enumerate(score):
        plt.plot(sizes, sc, label=f"Shift: {shifts[i]}")
    plt.legend()
    plt.show()


#===============================================================================
# Display Simplex noise conceptually
#===============================================================================
def local_correlation():
    save = True

    config.rows = 26
    config.landscape.params["size"] = 1

    force = UNI.yes_no("Force new connectivity matrix?", False)
    conn = ConnectivityMatrix(config).load(force=force)

    fig, ax = plt.subplots(num="simplex_noise")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plot_shift_arrows(conn.shift)
    bins = 6
    plt.xlim(-0.5, bins-0.5)
    plt.xticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plt.ylim(-0.5, bins-0.5)
    plt.yticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plt.tight_layout()
    if save:
        PIC.save_figure(fig.get_label(), fig, transparent=True)
    plt.show()


#===============================================================================
# METHODS
#===============================================================================

def plot_colored_shift(shift, note:str, save:bool=False):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift= shift.reshape((source, source))
    name = f"SHIFT_{note}"
    fig, ax = plt.subplots(
        num=name,
        # figsize=(4, 5),
        )
    plt.title("Categorical Shift")
    plt.xlabel("X-Position")
    plt.ylabel("Y-Position")
    # plt.xticks([10, 40, 70])
    # plt.yticks([10, 40, 70])
    im = plt.imshow(shift, origin="lower", cmap=plt.cm.hsv, vmax=DIRECTIONS)
    plt.colorbar(im,
                 orientation="horizontal")

    if save:
        PIC.save_figure(name, fig, sub_directory=config.sub_dir, transparent=True)


def calculate_direction(x, bins=DIRECTIONS, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v


def plot_shift(X=None, Y=None, D=None, name:str=None, **kwargs):
    # plt.figure(name, figsize=(4, 3))
    U, V = calculate_direction(D, **kwargs)
    plt.quiver(X, Y, U, V, pivot='middle', scale_units="xy", scale=1.125, units="dots", width=3)


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
        # plot_patch(center=(30, 17), radius=6, width=config.rows)
        # plot_patch(center=(36, 38), radius=6, width=config.rows)
        cbar.set_label("Indegree", rotation=270, labelpad=15)
        # ax.set_xticks([10, 40, 70])
        # ax.set_yticks([10, 40, 70])
        plt.tight_layout()

        if save:
            PIC.save_figure(name, fig, sub_directory=config.sub_dir, transparent=True)


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
