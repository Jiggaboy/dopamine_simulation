#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:08:08 2022

@author: hauke
"""

import cflogger
logger = cflogger.getLogger()

import numpy as np
import matplotlib.pyplot as plt
from lib import pickler as PIC

from plot.lib import remove_spines_and_ticks
from plot.figconfig import ConnectivityDistributionConfig as cfg



def main():
    joint_connectivity()
    plt.show()


def joint_connectivity(save:bool=True):
    logger.info("Start: Preparing figure of the connectivity distribution")
    shifts = (0, cfg.SHIFT)

    fig, ax = joint_connectivity_figure()
    neuron_handle = scatter_all_targets(ax)
    exc_handle_unshifted, inh_handle, full_dist_handle = hist_full_dist(shift=0, axis="x")
    exc_handle_shifted, _, _ = hist_full_dist(shift=cfg.SHIFT, axis="y")
    plt.legend([*full_dist_handle, *exc_handle_shifted, *exc_handle_unshifted, *inh_handle, *neuron_handle],
              ["joint", "exc. (shifted)", "exc. (unshifted)", "inh.", "pre-syn. neuron"])
    if save:
        PIC.save_figure(fig.get_label(), fig)


def joint_connectivity_figure():
    fig, ax = plt.subplots(**cfg.FIG_PARAMS)
    plt.title("Connectivity distributions")
    plot_scalebar(cfg.X_SCALEBAR, cfg.Y_SCALEBAR, cfg.WIDTH_SCALEBAR)
    remove_spines_and_ticks(ax)
    xylabel()
    return fig, ax


############# SCATTER TARGETS ###########################################################################################

def scatter_all_targets(ax:object, shift:float=cfg.SHIFT):
    """Plots all the targets and the neuron in the center."""
    logger.info("Scatter all targets")
    scatter_targets(ax, shift=0, color=cfg.C_TARGET, **cfg.targets)
    scatter_targets(ax, shift=shift, color=cfg.C_TARGET_SHIFTED, **cfg.targets)
    neuron_handle = plot_neuron(ax)
    return neuron_handle


def scatter_targets(ax:object, shift:float, center:np.ndarray, std:float, n_conn:int, color, **plot_kwargs):
    targets = np.random.normal(scale=std, size=(n_conn, 2))
    plt.plot(*(targets + center + shift).T, color=color, **plot_kwargs)


############# HISTOGRAM DISTRIBUTIONS ###################################################################################


def hist_exc_dist(shift:int=0, axis:str="x", std:float=cfg.STD, size=100*cfg.N_CONN, center=cfg.CENTER, **style):
    logger.info("Hist. of exc. distribution")
    dist, bins = get_hist_of_normal(center + shift, std, size=size)
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_inh_dist(axis:str="x", std:float=cfg.STD, size=100*cfg.N_CONN, center=cfg.CENTER, **style):
    logger.info("Hist. of inh. distribution")
    dist, bins = get_hist_of_normal(center, std * 2, size=size)
    dist /= 2
    handle = hist_dist(bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_full_dist(shift:int=0, axis:str="x", std:float=cfg.STD, size=100*cfg.N_CONN, center=cfg.CENTER):
    logger.info(f"Histogram distribution with shift {shift}.")
    exc_color = cfg.C_TARGET_SHIFTED if shift else cfg.C_TARGET
    exc_dist, bins, exc_handle = hist_exc_dist(shift=shift, axis=axis, color=exc_color)
    inh_dist, bins, inh_handle = hist_inh_dist(axis=axis, color=cfg.C_INH_HIST)
    mexican_hat = exc_dist - inh_dist
    full_dist_handle = hist_dist(bins, mexican_hat, axis=axis, lw=cfg.LW_DIST, color=cfg.C_FULL_HIST)
    return exc_handle, inh_handle, full_dist_handle


def hist_dist(bins, dist, axis:str="x", **style):
    """
    Plotting a histogram either on the x or on the y axis.
    """
    if axis == "x":
        data = bins[:-1], dist
    elif axis == "y":
        data = dist, bins[:-1]
    return plt.step(*data, **style)


########################################################################################################################


def get_exc_inh_distributions(position, std:float, size:int, shift=0):
    """
    Retrieves two normalized Gauss distributions.
    The exc. distribution has a std of {std}, whereas the inh. distribution has a std of {2 * std}, but normalizes to the half.
    """
    exc_dist, bins = get_hist_of_normal(position + shift, std, size=size)
    inh_dist, bins = get_hist_of_normal(position, 2*std, size=size)
    # Scale the inh. distribution
    inh_dist = inh_dist / 2
    return exc_dist, inh_dist, bins


def get_hist_of_normal(mean, std, size, normalize:bool=True):
    """
    Histograms a normal distribution in two dimensions.
    """
    dist = np.random.normal(mean, std, size=(size, 2))
    bins = np.arange(0, cfg.NROWS+1, cfg.BIN_WIDTH)
    dist_hist, _ = np.histogram(dist, bins)
    if normalize:
        dist_hist = normalize_histogram(dist_hist)
    return dist_hist, bins


def plot_neuron(ax:object):
    return ax.plot(*cfg.CENTER, marker="o", ms=cfg.NEURON_SIZE, ls="None", color=cfg.C_NEURON)


def normalize_histogram(distribution, prefactor=cfg.MAX_HIST):
    """Normalizes a histogram {distribution} such that the maximum value is the {prefactor}."""
    return prefactor * distribution / distribution.max()




def plot_scalebar(x:float, y:float, width:float):
    plt.plot([x, x+width], [y, y], color="black", linewidth=2)


def xylabel():
    plt.xlabel("unshifted (symmetric)")
    plt.ylabel("shifted (asymmetric)")


if __name__ == "__main__":
    main()
