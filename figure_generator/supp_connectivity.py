#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: 
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
from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors

from plot.constants import *

from lib.universal import dotdict
from plot.lib import remove_spines_and_ticks, remove_topright_spines
from lib.connectivitymatrix import ConnectivityMatrix
from params import config
import lib.pickler as PIC
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE
from plot.lib import add_colorbar_from_im
from plot.sequences import _get_sequence_landscape, truncate_colormap

from figure_generator.in_out_degree import calculate_direction, plot_shift_arrows
from figure_generator.figure1 import xyticks
#===============================================================================
# CONSTANTS
#===============================================================================
rcParams["font.size"] = 8
rcParams["figure.figsize"] = (17.6*cm, 6*cm)
rcParams["legend.fontsize"] = 7
rcParams["legend.markerscale"] = 0.6
rcParams["legend.handlelength"] = 1.25
rcParams["legend.columnspacing"] = 1
rcParams["legend.handletextpad"] = 1
rcParams["legend.labelspacing"] = .1
rcParams["legend.borderpad"] = .25
rcParams["legend.handletextpad"] = .5
rcParams["legend.framealpha"] = 1
rcParams["axes.labelpad"] = 2

cmap_conn = plt.cm.PRGn
cmap_base = plt.cm.PRGn

cticks = (-100, -50, 0, 25)
nticks = (0, 5000, 10000)

filename = "supp_connectivity"
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1, 1))
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.16,
        top=0.86,
        wspace=0.4,
        hspace=0.0,
    )
    
    conn = ConnectivityMatrix(config)
    vmin = conn.connectivity_matrix.min()*0.9
    vmax = conn.connectivity_matrix.max()*0.9
    # norm = colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax)
    # zero_pos = norm(0.0)
    # cmap_conn = colors.LinearSegmentedColormap.from_list(
    #     "PRGn_asym",
    #     [
    #         (0.0, cmap_base(0.)),
    #         (zero_pos, "white"),
    #         (1.0, cmap_base(1.)),
    #     ]
    # )
    norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0.0,)
    # W = conn.connections
    ax = fig.add_subplot(gs[:, 0])
    ax.set(xlabel="Post-synaptic neuron ID", ylabel="Pre-synaptic neuron ID", title="Connectivity Matrix",
           xticks=nticks, yticks=nticks)
    im = ax.imshow(conn.connectivity_matrix[:, :], cmap=cmap_conn, 
    # im = ax.imshow(conn.connectivity_matrix[9000:11000, 9000:11000], cmap=cmap_conn, 
                   norm=norm, )
    
    inset_idx = 8000, 8150
    index_slice = slice(*inset_idx)
    Z2 = conn.connectivity_matrix[index_slice, index_slice]
    
    # ax.imshow(Z2, origin="lower")

    # inset Axes....
    x1, x2, y1, y2 = inset_idx[0], inset_idx[1], inset_idx[1], inset_idx[0] 
    axins = ax.inset_axes(
        [0.4, 0.5, 0.43, 0.43],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.tick_params("both", which="both", left=False, bottom=False)
    axins.imshow(Z2, norm=norm, cmap=cmap_conn, extent=(x1, x2, y1, y2))

    ax.indicate_inset_zoom(axins, edgecolor="black")
    cbar = fig.colorbar(im)
    # cbar.set_ticks((-1e2, -1e1, 0, 1e1))
    cbar.set_ticks(cticks)
    cbar.set_label("Connection strength", rotation=-90, labelpad=8)

    ax = fig.add_subplot(gs[:, 1])
    remove_topright_spines(ax)
    ax.set(yscale="log", xlabel="Connection strength", ylabel="Occurence", title="Connection Strengths",
           xticks=cticks)
    ax.tick_params("y", which="minor", left=False)
    W = conn.connectivity_matrix[np.nonzero(conn.connectivity_matrix)]
    
    n, bins, patches = ax.hist(W.flatten(), bins=28)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    col = cmap_conn(norm(bin_centers))
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', c)
        
    PIC.save_figure(filename, fig)
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    # plt.show()