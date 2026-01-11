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

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from plot.constants import *

from lib.universal import dotdict
from plot.lib import remove_spines_and_ticks
from lib.connectivitymatrix import ConnectivityMatrix
from params import config
import lib.pickler as PIC
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE


from figure_generator.in_out_degree import calculate_direction, plot_shift_arrows

#===============================================================================
# CONSTANTS
#===============================================================================

rcParams["font.size"] = 12
rcParams["figure.figsize"] = (18*cm, 8*cm)

# PANEL NETWORK LAYOUT
side_length = 6

# PANEL CONNECTIVITY DISTRIBUTION
nrows = 40
MAX_HIST = 8
BIN_WIDTH = 1


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    outer = [['upper left',  "upper center", "upper right"],
              ['lower left', "lower center", 'lower right']]

    fig, axd = plt.subplot_mosaic(outer, layout="constrained")
    print(axd)
    for k, ax in axd.items():
        annotate_axes(ax, f'axd[{k!r}]')
        print(k, ax)

    panel_network_layout(axd["upper left"], side_length)

    panel_connectivity_distribution(axd["upper center"])

    panel_simplex_noise(axd["upper right"], config)

    panel_indegree(axd["lower left"], config)

    panel_avg_activity(axd["lower center"], config)

    panel_STAS(axd["lower right"])
    # fig.savefig(filename + ".png", transparent=True)
    # fig.savefig(filename + ".svg", transparent=True)


#===============================================================================
# METHODS
#===============================================================================
def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")


def panel_network_layout(ax:object, side_length:int):
    pos = np.arange(side_length)
    x_pos_exc, y_pos_exc = np.meshgrid(pos, pos)


    iside = np.arange(0.5, side_length, 2)
    x_pos_inh, y_pos_inh = np.meshgrid(iside, iside)

    ax.scatter(x_pos_exc, y_pos_exc, color=KTH_PINK, marker="o", label="exc. neuron")
    ax.scatter(x_pos_inh, y_pos_inh, color=KTH_BLUE, marker="x", label="inh. neuron")
    ax.legend(loc="upper right")
    # ax.set_title("Layout of exc. (red dots) and \ninh. (blue crosses) neurons")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(-0.5, side_length-0.5)
    ax.set_xticks(np.arange(0, side_length), ["...", *np.arange(20, 20 + side_length-2), "..."])
    ax.set_ylim(-0.5, side_length-0.5)
    ax.set_yticks(np.arange(0, side_length), ["...", *np.arange(20, 20 + side_length-2), "..."])


def panel_connectivity_distribution(ax:object):
    # Network
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


    #### SCALEBAR
    X_SCALEBAR = 8
    Y_SCALEBAR = 35
    WIDTH_SCALEBAR = 2

    scalebar_style = {"color": "black", "linewidth": 2}
    remove_spines_and_ticks(ax)

    ax.set_xlabel("static (symmetric)")
    ax.set_ylabel("shifted (asymmetric)")
    ax.set_xlim(0, nrows)
    ax.set_ylim(0, nrows)

    logger.info("Scatter shifted and unshifted targets.")
    scatter_targets(ax, center=center, shift=0, color=C_TARGET, **targets, **target_style)
    scatter_targets(ax, center=center, shift=[SHIFT, -SHIFT], color=C_TARGET_SHIFTED, **targets, **target_style)
    neuron = ax.plot(*center, **neuron_style)

    logger.info("Histogram of the static targets.")
    hist_params = dict(**targets, **{
        "axis": "x",
        })
    hist_params["n_conn"] *= 250

    logger.info("Hist. of exc. distribution")
    exc_dist, bins = get_hist_of_normal(center, std=targets["std"], size=targets["n_conn"])
    exc_handle_unshifted = hist_dist(ax, bins, exc_dist, color=C_TARGET, axis="x")

    # exc_dist, bins, exc_handle_unshifted = hist_exc_dist(ax, shift=0., center=center, color=C_TARGET, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(ax, center=center, color=C_INH_HIST, factor=stdIE_ratio, **hist_params)
    joint_dist = hist_joint_dist(ax, exc_dist, inh_dist, bins, axis=hist_params["axis"], color=C_FULL_HIST)

    logger.info("Histogram of the shifted targets.")
    hist_params["axis"] = "y"

    exc_dist_shifted, bins, exc_handle_shifted = hist_exc_dist(ax, shift=-SHIFT, center=center, color=C_TARGET_SHIFTED, **hist_params)
    inh_dist, bins, inh_handle = hist_inh_dist(ax, center=center, color=C_INH_HIST, factor=stdIE_ratio, **hist_params)
    joint_dist = hist_joint_dist(ax, exc_dist_shifted, inh_dist, bins, axis=hist_params["axis"], color=C_FULL_HIST)


    ax.legend([*exc_handle_shifted, *exc_handle_unshifted, *inh_handle, *neuron],
              ["exc. (shifted)", "exc. (static)", "inh.", "presyn. neuron"],
              fontsize="small",
               # scatteryoffsets=[0.5],
               # frameon=False,
               loc="upper right",
               labelspacing=.2,
              )

    ax.axhline(4, xmin=0.1, ls="--", c="k", zorder=-5)
    ax.axvline(4, ymin=0.1, ls="--", c="k", zorder=-5)
    ax.axhline(nrows // 2 - SHIFT, xmin=0.1, xmax=0.575, ls="--", c=KTH_GREEN, zorder=5)
    ax.axvline(nrows // 2, ymin=0.1, ymax=0.575, ls="--", c=KTH_YELLOW, zorder=5)


def panel_simplex_noise(ax:object, config:object):
    tmp_config = copy.copy(config)
    tmp_config.landscape = copy.copy(config.landscape)
    tmp_config.landscape.params = dict(config.landscape.params)

    tmp_config.rows = 26
    tmp_config.landscape.params["size"] = 2
    tmp_config.landscape.params["base"] = 8

    conn = ConnectivityMatrix(tmp_config)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    bins = 6
    ax.set_xlim(-0.5, bins-0.5)
    ax.set_xticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    ax.set_ylim(-0.5, bins-0.5)
    ax.set_yticks(np.arange(0, bins), ["...", *np.arange(20, 20 + bins-2), "..."])
    plot_shift_arrows(conn.shift, ax=ax)


def panel_indegree(ax:object, config:object):
    conn = ConnectivityMatrix(config)
    indegree, _ = conn.degree(conn._EE)
    indegree = indegree * config.synapse.weight

    degree_cmap = plt.cm.jet
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([10, 40, 70])
    ax.set_yticks([10, 40, 70])
    im = ax.imshow(indegree,
                    origin="lower",
                    cmap=degree_cmap,
                    # vmin=550, vmax=950,
    )
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # plt.colorbar(im, cax=cax)

    cbar = plt.colorbar(im,
                orientation="vertical",
                ticks = [600, 750, 900],
                cax=cax
                )
    cbar.set_label("In-degree", rotation=270, labelpad=15)


def panel_avg_activity(ax:object, config:object):
    tags = config.baseline_tags

    # Gather all rates
    rates = []
    for tag in tags:
        logger.info(f"Load {tag}...")
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
        rates.append(avgRate)
    rates = np.asarray(rates)

    # Average if more than one run
    if rates.ndim > 1:
        rates = rates.mean(axis=0)

    norm = (0, 0.5)
    cmap = COLOR_MAP_ACTIVITY

    create_image(rates, norm, cmap, axis=ax)

    cbar = add_colorbar(ax, norm, cmap)
    cbar.set_label("Avg. activity [a.u.]", rotation=270, labelpad=15)
    cbar.set_ticks([0.0, 0.2, 0.4])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([10, 40, 70])
    ax.set_yticks([10, 40, 70])


def panel_STAS(ax:object):
    remove_spines_and_ticks(ax)

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

def hist_exc_dist(ax:object, shift:int, axis:str, std:float, n_conn:int, center=(0, 0), **style):
    logger.info("Hist. of exc. distribution")
    dist, bins = get_hist_of_normal(center + shift, std, size=n_conn)
    handle = hist_dist(ax, bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_inh_dist(ax:object, axis:str, std:float, n_conn:int, center=(0, 0), factor:int=2., **style):
    logger.info("Hist. of inh. distribution")
    dist, bins = get_hist_of_normal(center, std * factor, size=n_conn)
    dist /= factor
    handle = hist_dist(ax, bins, dist, axis=axis, **style)
    return dist, bins, handle


def hist_joint_dist(ax:object, exc_dist, inh_dist, bins, **hist_params):
    return hist_dist(ax, bins, exc_dist - inh_dist, **hist_params)



def hist_dist(ax:object, bins, dist, axis:str="x", **style):
    """
    Plotting a histogram either on the x or on the y axis.
    """
    offset = 4
    dist = dist + offset
    if axis == "x":
        data = bins[1:], dist
    elif axis == "y":
        data = dist, bins[:-1]
    return ax.step(*data, **style)


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


def plot_scalebar(x:float, y:float, width:float, **scalebar_style):
    plt.plot([x, x+width], [y, y], **scalebar_style)







if __name__ == '__main__':
    main()
    plt.show()
