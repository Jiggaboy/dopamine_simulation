#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:08:08 2022

@author: hauke
"""


import numpy as np
import matplotlib.pyplot as plt

import lib.lcrn_network as lcrn

print("Start: Preparing figure of the connectivity distribution")

nrows = 70
std = 5
n_conn = int(nrows * nrows * .2)
position = np.asarray((35, 35), dtype=int)
SHIFT = 4
MARKER = "."
MAX_HIST = 12
BIN_WIDTH = 1

FIGSIZE = (4.2, 4.2)
SAVE_FIG_PATH = "poster/{}.svg"
AXIS_MARGIN = 5

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
c_unshifted = color_cycle[0]
c_shifted = color_cycle[2]


KTH_GREEN = 176, 201, 43
KTH_PINK = 216, 84, 151
KTH_GREY = 101, 101, 108

colors = KTH_GREEN, KTH_PINK, KTH_GREY
colors = np.asarray(colors) / 255

c_exc = colors[0]
c_inh = colors[1]
c_mexican = colors[2]
c_neuron = "black"

NEURON_SIZE = 14
SPINE_WIDTH = 4



def main():

    titles = ("unshifted", "shifted")
    shifts = (0, SHIFT)
    for t, s in zip(titles, shifts):
        figure_with_xy_axes(t)
        plt.title("Connectivity distribution")
        scatter_exc_targets(std, size=n_conn, shift=s)
        hist_connectitivy_distributions(position, std, size=100 * n_conn, shift=s)

        plt.plot([55, 56], [10, 10], color="black", linewidth=3)

        set_layout()
        plt.savefig(SAVE_FIG_PATH.format(t))


def figure_with_xy_axes(title):
    plt.figure(title, figsize=FIGSIZE)
    # plt.axhline(color="k")
    # plt.axvline(color="k")


def set_layout(rows:int=nrows, margin:float=AXIS_MARGIN):
    ax = plt.gca()

    PY_OFFSET = -.5
    lim = PY_OFFSET - margin, PY_OFFSET + rows + margin
    plt.xlim(*lim)
    plt.ylim(*lim)
    ticks = np.linspace(0, rows, 3, endpoint=True)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticks([])
    ax.set_yticks([])

    tick_params = {"width": SPINE_WIDTH, "length": SPINE_WIDTH * 3, "labelleft": False, "labelbottom": False}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        # ax.spines[s].set_visible(False)
        ax.spines[s].set_linewidth(SPINE_WIDTH)
    for s in ('bottom', 'left'):
        # ax.spines[s].set_visible(False)
        ax.spines[s].set_linewidth(SPINE_WIDTH)

    plt.tight_layout()


def hist_connectitivy_distributions(position:np.ndarray, std:float, size:int, shift:float=0):
    axes = ("x", "y")

    for pos, axis in zip(np.arange(2), axes):
        exc_dist, inh_dist, bins = get_exc_inh_distributions(position[pos], std, size=size, shift=shift)
        mexican_hat = (exc_dist) - inh_dist
        plot_hist_dist(bins[:-1], exc_dist, inh_dist, mexican_hat, axis=axis)



def scatter_exc_targets(std:float, size:int, shift:float=0):
    exc_targets = np.random.normal(scale=std, size=(size, 2))
    plt.plot(*(exc_targets + position + shift).T, color=c_exc, marker=MARKER, linestyle="None", ms=4)
    plt.plot(*position, marker="o", color=c_neuron, ms=NEURON_SIZE)


def plot_hist_dist(bins, exc_dist, inh_dist, full_dist=None, axis:str="x"):
    colors = (c_exc, c_inh, c_mexican)
    dists = [exc_dist, inh_dist]
    if full_dist is not None:
        dists.append(full_dist)

    if axis == "x":
        for dist, color in zip(dists, colors):
            plt.step(bins, dist, color=color)
    elif axis == "y":
        for dist, color in zip(dists, colors):
            plt.step(dist, bins, color=color)




def get_exc_inh_distributions(position, std:float, size:int, shift=0):
    """Retrieves two normalized Gauss distributions.
    The exc. distribution has a std of {std}, whereas the inh. distribution has a std of {2 * std}, but normalizes to the half.
    """

    exc_dist, bins = get_hist_of_normal(position + shift, std, size=size)
    inh_dist, bins = get_hist_of_normal(position, 2*std, size=size)
    # Scale the inh. distribution
    inh_dist = inh_dist / 2

    return exc_dist, inh_dist, bins


def get_hist_of_normal(mean, std, size, normalize:bool=True):
    dist = np.random.normal(mean, std, size=size)
    bins = np.arange(0, nrows+1, BIN_WIDTH)
    dist_hist, _ = np.histogram(dist, bins)
    if normalize:
        dist_hist = normalize_histogram(dist_hist)
    return dist_hist, bins




def normalize_histogram(distribution, prefactor=MAX_HIST):
    """Normalizes a histogram {distribution} such that the maximum value is the {prefactor}."""
    return MAX_HIST * distribution / distribution.max()

if __name__ == "__main__":
    main()
