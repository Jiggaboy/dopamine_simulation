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
from plot.constants import KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE



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

## Targets
targets = dotdict({
    "std": 2.5,
    "n_conn": 200,
})
target_style = {
    "marker": "o",
    "ms": 2.25,
    "linestyle": "None",
}

SHIFT = 2

# Targets (style)
MARKER = "."
C_TARGET = KTH_PINK
C_TARGET_SHIFTED = KTH_GREEN

neuron_style = {
    "ms": 12,
    "color": "black",
    "marker": "o",
    "ls": "None",
}


#### STYLE HISTOGRAMS
C_INH_HIST = KTH_BLUE
C_FULL_HIST = KTH_GREY

MAX_HIST = 10
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

    plot_scalebar(X_SCALEBAR, Y_SCALEBAR, WIDTH_SCALEBAR, **scalebar_style)
    remove_spines_and_ticks(ax)

    ax.set_xlabel("unshifted (symmetric)", fontsize=12)
    ax.set_ylabel("shifted (asymmetric)")

    logger.info("Scatter shifted and unshifted targets.")
    scatter_targets(ax, center=center, shift=0, color=C_TARGET, **targets, **target_style)
    scatter_targets(ax, center=center, shift=[SHIFT, -SHIFT], color=C_TARGET_SHIFTED, **targets, **target_style)
    neuron = plot_neuron(ax, **neuron_style)

    logger.info("Histogram of the unshifted targets.")
    hist_params = dict(**targets, **{
        "axis": "x",
        })
    hist_params["n_conn"] *= 10

    exc_dist, bins, exc_handle_unshifted = hist_exc_dist(shift=0., color=C_TARGET, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(color=C_INH_HIST, **hist_params)
    joint_dist = hist_joint_dist(exc_dist, inh_dist, bins, axis=hist_params["axis"], color=C_FULL_HIST)

    logger.info("Histogram of the shifted targets.")
    hist_params["axis"] = "y"

    exc_dist, bins, exc_handle_shifted = hist_exc_dist(shift=-SHIFT, color=C_TARGET_SHIFTED, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(color=C_INH_HIST, **hist_params)
    joint_dist = hist_joint_dist(exc_dist, inh_dist, bins, axis=hist_params["axis"], color=C_FULL_HIST)


    plt.legend([*joint_dist, *exc_handle_shifted, *exc_handle_unshifted, *inh_handle, *neuron],
              ["joint", "exc. (shifted)", "exc. (unshifted)", "inh.", "pre-syn. neuron"],
              fontsize="small",
               # scatteryoffsets=[0.5],
               # frameon=False,
               loc="upper right",
               labelspacing=.25,
              )
    if save:
        PIC.save_figure(fig.get_label(), fig, transparent=True)
    plt.show()


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


def hist_inh_dist(axis:str, std:float, n_conn:int, center=center, **style):
    logger.info("Hist. of inh. distribution")
    dist, bins = get_hist_of_normal(center, std * 2, size=n_conn)
    dist /= 2
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_joint_dist(exc_dist, inh_dist, bins, **hist_params):
    return hist_dist(bins, exc_dist - inh_dist, **hist_params)



def hist_dist(bins, dist, axis:str="x", **style):
    """
    Plotting a histogram either on the x or on the y axis.
    """
    if axis == "x":
        data = bins[:-1], dist
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
