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
from matplotlib import rcParams

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


cmap_count = plt.cm.hot_r
cmap_grad = plt.cm.seismic
seeds = np.arange(8)

filename = "supp_sequence_count_and_gradient"
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1, 1))
    fig.subplots_adjust(
        left=0.0,
        right=0.9,
        bottom=0.18,
        top=0.84,
        wspace=0.0,
        hspace=0.0,
    )
    
    
    seq_counts = np.zeros((seeds.size, config.rows, config.rows), dtype=int)
    for s, seed in enumerate(seeds):
        tag = config.baseline_tag(seed=seed)
        spikes, labels = PIC.load_spike_train(tag, config)
        seq_count = _get_sequence_landscape(spikes, labels, config.rows)
        seq_counts[s] = seq_count
    
    ax = fig.add_subplot(gs[:, 0])
    im = create_image(seq_counts.mean(axis=0).T, cmap=cmap_count, axis=ax)
    ax.set(
        xticks=xyticks, yticks=xyticks,
        xlabel="X", ylabel="Y", title="Sequence Counts")
    cbar = add_colorbar_from_im(ax, im)
    cbar.set_ticks(np.linspace(0, 30, 4, dtype=int))
    cbar.set_label("Seq. count", rotation=270, labelpad=10)
    
    
    ax_grad = fig.add_subplot(gs[:, 1])
    ax_grad.set(
        xticks=xyticks, yticks=xyticks,
        xlabel="X", ylabel="Y", title="Sequence Counts Gradient")
    grad_seq_x, grad_seq_y = np.gradient(seq_count, edge_order=2)
    grad = np.stack((grad_seq_x, grad_seq_y))
    cmap = truncate_colormap(cmap_grad, minval=0.5, maxval=0.95)
    norm = (0, 12)
    grad_norm = np.linalg.norm(grad, axis=0)
    im = create_image(grad_norm.T, cmap=cmap, axis=ax_grad, norm=norm)
    cbar = add_colorbar(ax_grad, norm, cmap, ticks=[0, 5, 10, ])
    cbar.set_label(r"$\nabla$ Sequence count", rotation=270, labelpad=15)

    PIC.save_figure(filename, fig)
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
