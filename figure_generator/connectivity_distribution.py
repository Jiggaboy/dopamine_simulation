#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the connectivity distribution as histograms and individual targets in the shifted and unshifted case.
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.3'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import lib.pickler as PIC

from plot.lib import remove_spines_and_ticks

from lib.universal import dotdict
from plot.constants import *



#===============================================================================
# Params
#===============================================================================

save = True
rcParams["font.size"] = 12
rcParams["figure.figsize"] = (3.5, 3.5)

# Figure
fig_params = {
    "num": "joint_connectivity",
    # "figsize": (4.4, 4.)
}

# Network
nrows = 40
center = np.asarray((nrows // 2, nrows // 2), dtype=int)
stdIE_ratio = 6 / 2.75

## Targets
targets = dotdict({
    "std": 2.75,
    "n_conn": 200,
})
target_style = {
    "marker": "o",
    "ms": 2.4,
    "linestyle": "None",
    "mec": "k",
    "mew": 0.2,
}

SHIFT = 2

# Targets (style)
MARKER = "."
C_TARGET = KTH_YELLOW
C_TARGET_SHIFTED = KTH_GREEN

neuron_style = {
    "ms": 10,
    "color": KTH_PINK,
    "marker": "o",
    "ls": "None",
    "mec": "k",
    "zorder": 10,
}


#### STYLE HISTOGRAMS
C_INH_HIST = KTH_BLUE
C_FULL_HIST = KTH_GREY

MAX_HIST = 8
BIN_WIDTH = 1


#### SCALEBAR
X_SCALEBAR = 8
Y_SCALEBAR = 35
WIDTH_SCALEBAR = 2

scalebar_style = {"color": "black", "linewidth": 2}


#===============================================================================
# MAIN
#===============================================================================

def main():
    logger.info("Start: Preparing figure of the connectivity distribution")
    """Sets the features: title, xlabel, ylabel. Removes the spines."""
    fig, ax = plt.subplots(tight_layout=False, **fig_params)
    # fig.suptitle("Connectivity kernel", fontsize="xx-large")

    # plot_scalebar(X_SCALEBAR, Y_SCALEBAR, WIDTH_SCALEBAR, **scalebar_style)
    remove_spines_and_ticks(ax)

    ax.set_xlabel("static (symmetric)")
    ax.set_ylabel("shifted (asymmetric)")
    ax.set_xlim(0, nrows)
    ax.set_ylim(0, nrows)

    logger.info("Scatter shifted and unshifted targets.")
    scatter_targets(ax, center=center, shift=0, color=C_TARGET, **targets, **target_style)
    scatter_targets(ax, center=center, shift=[SHIFT, -SHIFT], color=C_TARGET_SHIFTED, **targets, **target_style)
    neuron = plot_neuron(ax, **neuron_style)

    logger.info("Histogram of the static targets.")
    hist_params = dict(**targets, **{
        "axis": "x",
        })
    hist_params["n_conn"] *= 250

    exc_dist, bins, exc_handle_unshifted = hist_exc_dist(shift=0., color=C_TARGET, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(color=C_INH_HIST, factor=stdIE_ratio, **hist_params)
    joint_dist = hist_joint_dist(exc_dist, inh_dist, bins, axis=hist_params["axis"], color=C_FULL_HIST)

    logger.info("Histogram of the shifted targets.")
    hist_params["axis"] = "y"

    exc_dist_shifted, bins, exc_handle_shifted = hist_exc_dist(shift=-SHIFT, color=C_TARGET_SHIFTED, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(color=C_INH_HIST, factor=stdIE_ratio, **hist_params)
    joint_dist = hist_joint_dist(exc_dist_shifted, inh_dist, bins, axis=hist_params["axis"], color=C_FULL_HIST)


    plt.legend([*exc_handle_shifted, *exc_handle_unshifted, *inh_handle, *neuron],
              ["exc. (shifted)", "exc. (static)", "inh.", "presyn. neuron"],
              fontsize="small",
               # scatteryoffsets=[0.5],
               # frameon=False,
               loc="upper right",
               labelspacing=.2,
              )

    plt.axhline(4, xmin=0.1, ls="--", c="k", zorder=-5)
    plt.axvline(4, ymin=0.1, ls="--", c="k", zorder=-5)
    plt.axhline(nrows // 2 - SHIFT, xmin=0.1, xmax=0.575, ls="--", c=KTH_GREEN, zorder=5)
    plt.axvline(nrows // 2, ymin=0.1, ymax=0.575, ls="--", c=KTH_YELLOW, zorder=5)
    if save:
        PIC.save_figure(fig.get_label(), fig, transparent=True)


#===============================================================================
# SCATTER TARGETS
#===============================================================================

def scatter_targets(ax:object, shift:float, center:np.ndarray, std:float, n_conn:int, **plot_kwargs):
    """Finds targets and scatter them."""
    targets = np.random.normal(scale=std, size=(n_conn, 2))
    ax.plot(*(targets + center + shift).T, **plot_kwargs)


#===============================================================================
# HISTORGRAM DISTRIBUTIONS
#===============================================================================

def hist_exc_dist(shift:int, axis:str, std:float, n_conn:int, center=center, **style):
    logger.info("Hist. of exc. distribution")
    dist, bins = get_hist_of_normal(center + shift, std, size=n_conn)
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_inh_dist(axis:str, std:float, n_conn:int, center=center, factor:int=2., **style):
    logger.info("Hist. of inh. distribution")
    dist, bins = get_hist_of_normal(center, std * factor, size=n_conn)
    dist /= factor
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_joint_dist(exc_dist, inh_dist, bins, **hist_params):
    return hist_dist(bins, exc_dist - inh_dist, **hist_params)



def hist_dist(bins, dist, axis:str="x", **style):
    """
    Plotting a histogram either on the x or on the y axis.
    """
    offset = 4
    dist = dist + offset
    # dist = dist[offset:-offset]
    # bins = bins[offset:-offset]
    if axis == "x":
        data = bins[1:], dist
    elif axis == "y":
        data = dist, bins[:-1]
    return plt.step(*data, **style)


def get_hist_of_normal(mean, std, size, normalize:bool=True):
    """
    Histograms a normal distribution in two dimensions.
    """
    dist = np.random.normal(mean, std, size=(size, 2))
    bins = np.arange(0, nrows+1, BIN_WIDTH)
    dist_hist, _ = np.histogram(dist, bins)
    if normalize:
        dist_hist = normalize_histogram(dist_hist)
    return dist_hist, bins



def normalize_histogram(distribution, prefactor=MAX_HIST):
    """Normalizes a histogram {distribution} such that the maximum value is the {prefactor}."""
    return prefactor * distribution / distribution.max()


def plot_neuron(ax:object, **style):
    return ax.plot(*center, **style)

def plot_scalebar(x:float, y:float, width:float, **scalebar_style):
    plt.plot([x, x+width], [y, y], **scalebar_style)




if __name__ == "__main__":
    main()
    plt.show()
