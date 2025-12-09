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
from lib.connectivitymatrix import ConnectivityMatrix, CustomConnectivityMatrix
from plot.lib import plot_patch
import lib.universal as UNI

DIRECTIONS = 8
HIST_DEGREE = False

degree_num_kwargs = {
    "figsize": (5.5, 3.5),
    # "tight_layout": True,
}

rcParams["font.size"] = 12
rcParams["figure.figsize"] = (5.5, 3.5)


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    # seed_range = 5
    # conns = np.zeros((seed_range, config.no_exc_neurons))
    # for s in range(seed_range):
    #     config.landscape.seed = s
    #     conn = ConnectivityMatrix(config)
    #     degrees = conn.degree(conn._EE)
    #     conns[s] = degrees[0].flatten()
    # plt.figure("Indegree hist")
    # plt.title("Indegree across landscape seeds")
    # plt.xlabel("Indegree (# of EE connections)")
    # plt.ylabel("occurrence")
    # plt.hist(conns.T, bins=25)
    # plt.show()
    # return


    force = UNI.yes_no("Force new connectivity matrix?")
    # conn = ConnectivityMatrix(config, force=force)
    conn = CustomConnectivityMatrix(config, force=force)
    plot_colored_shift(conn.shift, note=f"{config.landscape.shift}_{config.landscape.params['size']}", save=False)
    # plot_shift_arrows(conn.shift)

    ######### CURL #####################################


    def torus_gradient(x, buffer:int=8, axis=None):
        if isinstance(axis, tuple):
            from functools import partial
            func = partial(torus_gradient, x, buffer)
            return tuple(map(func, axis))
        x_grad = np.gradient(x, axis=axis, edge_order=2)
        x_grad_shifted = np.gradient(np.roll(x, buffer, axis=(0, 1)), axis=axis, edge_order=2)
        x_grad_shifted = np.roll(x_grad_shifted, -buffer, axis=(0, 1))
        x_grad[:buffer] = x_grad_shifted[:buffer]
        x_grad[-buffer:] = x_grad_shifted[-buffer:]
        x_grad[:, :buffer] = x_grad_shifted[:, :buffer]
        x_grad[:, -buffer:] = x_grad_shifted[:, -buffer:]
        return x_grad
    # angle_dy, angle_dx = torus_gradient(angles, axis=(0, 1))


    shift = np.reshape(conn.shift, (config.rows, config.rows))
    d1, d2 = calculate_direction(shift, bins=DIRECTIONS)
    angles = np.arctan2(d2, d1)

    # d1_dx, d1_dy = np.gradient(d1, axis=(0, 1))
    # d2_dx, d2_dy = np.gradient(d2, axis=(0, 1))


    # dx = d1_dx + d2_dx
    # dy = d1_dy + d2_dy


    angle_dy, angle_dx = np.gradient(angles, axis=(0, 1))
    # Convert to x and y
    angle_dx_x = np.cos(angle_dx)
    angle_dx_y = np.sin(angle_dx)

    angle_dy_x = np.cos(angle_dy)
    angle_dy_y = np.sin(angle_dy)

    angles = np.arctan2(angle_dx_y+angle_dy_y, angle_dx_x+angle_dy_x)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(angle_dx_x, origin="lower")
    # plt.colorbar()
    axes[0, 1].imshow(angle_dx_y, origin="lower")
    # plt.colorbar()
    axes[1, 0].imshow(angle_dy_x, origin="lower")
    # plt.colorbar()
    axes[1, 1].imshow(angle_dy_y, origin="lower")
    plt.figure("angles")
    plt.imshow(angles, origin="lower")
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(d1, origin="lower")
    axes[1].imshow(d2, origin="lower")

    # plt.colorbar()
    plt.show()
    quit()

    angle_dy, angle_dx = np.gradient(angles, axis=(0, 1))

    div = angle_dx #+ angle_dy
    plt.figure()
    plt.imshow(div, origin="lower")



    plt.colorbar()
    plt.show()
    quit()


    angle_dx[angle_dx > np.pi] = angle_dx[angle_dx > np.pi] - 2*np.pi
    angle_dx[angle_dx <-np.pi] = angle_dx[angle_dx <-np.pi] + 2*np.pi
    angle_dy[angle_dy > np.pi] = angle_dy[angle_dy > np.pi] - 2*np.pi
    angle_dy[angle_dy <-np.pi] = angle_dy[angle_dy <-np.pi] + 2*np.pi

    angle_grad = np.arctan2(angle_dy, angle_dx)

    angle_grad_grad = torus_gradient(angle_grad, axis=(0, 1))
    angle_grad_dx, angle_grad_dy = tuple(angle_grad_grad)

    angle_grad_dx[angle_grad_dx > np.pi] = angle_grad_dx[angle_grad_dx > np.pi] - 2*np.pi
    angle_grad_dx[angle_grad_dx <-np.pi] = angle_grad_dx[angle_grad_dx <-np.pi] + 2*np.pi
    angle_grad_dy[angle_grad_dy > np.pi] = angle_grad_dy[angle_grad_dy > np.pi] - 2*np.pi
    angle_grad_dy[angle_grad_dy <-np.pi] = angle_grad_dy[angle_grad_dy <-np.pi] + 2*np.pi


    x = np.arange(config.rows)
    y = np.arange(config.rows)
    X, Y = np.meshgrid(x, y)
    # Compute the curl (only z-component in 2D)
    # curl = np.gradient(V, x, axis=1) - np.gradient(U, y, axis=0)

    # Plot the vector field
    plt.figure("curl")
    plt.imshow(angles, origin="lower", vmin=-np.pi, vmax=np.pi, cmap="hsv")
    plt.colorbar()
    plt.quiver(X, Y, angle_dx, angle_dy, color='white')
    # plt.quiver(X, Y, angle_grad_dx, angle_grad_dy, color='white')
    # plt.contourf(X, Y, curl, 50, cmap='coolwarm', alpha=0.6)
    # plt.colorbar()
    plt.title('Curl of Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


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
        plot_degree(degrees[0], note=f"avg_degree_{config.landscape.shift}_{n}_{config.landscape.params['size']}", save=True, config=config)

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
            # plot_patch(c, config.radius, width=config.rows)
        # _, ax = plot_degree(degrees[0], note=f"{config.landscape.shift}_{n}", save=False, config=config)
        # for c, center in enumerate(config.center_range.values()):
        #     print(center)
            # plot_patch(center, config.radius[0], width=config.rows, axis=ax)


        break
    # plot_scaled_indegree(conn, config=config)


#===============================================================================
# METHODS
#===============================================================================

def plot_colored_shift(shift, note:str, save:bool=False):
    if len(shift.shape) < 2:
        source = np.sqrt(shift.size).astype(int)
        shift  = shift.reshape((source, source))
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
    plt.colorbar(im, orientation="horizontal")

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
        cbar.set_label("In-degree", rotation=270, labelpad=15)
        # ax.set_xticks([10, 40, 70])
        # ax.set_yticks([10, 40, 70])
        # plt.tight_layout()

        if save:
            PIC.save_figure(name, fig, sub_directory=config.sub_dir, transparent=True)
        return fig, ax

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
    plt.show()
