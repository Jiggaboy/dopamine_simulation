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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from skimage.morphology import max_tree
import networkx as nx
from dataclasses import dataclass

from params import config
import lib.universal as UNI
import lib.pickler as PIC
from plot.lib.frame import create_image, create_images_on_axes
from plot.lib.basic import add_colorbar, plot_patch_from_tag, add_colorbar_from_im, remove_spines_and_ticks,remove_topright_spines
from plot.constants import cm
from plot.sequences import _get_sequence_landscape
from lib.connectivitymatrix import ConnectivityMatrix

from figure_generator.figure1 import indegree_low, indegree_high

from tree import grow_forest
#===============================================================================
# CONSTANTS
#===============================================================================
rcParams["font.size"] = 8
rcParams["figure.figsize"] = (17.6*cm, 6*cm)
rcParams["legend.fontsize"] = 7
rcParams["legend.framealpha"] = 1
rcParams["axes.labelpad"] = 2

filename = "transmissive_neurons"

example_seed = 3
seeds = np.arange(8)
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=4, width_ratios=(1.2, 1, .4, 1), height_ratios=(1, 1.3))
    fig.subplots_adjust(
        left=0.0,
        right=0.99,
        bottom=0.18,
        top=0.84,
        wspace=0.0,
        hspace=0.0,
    )
    
    cmap = plt.cm.hot_r
    ax = fig.add_subplot(gs[:, 0], projection= "3d")
    
    tag = config.baseline_tag(seed=example_seed)
    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = _get_sequence_landscape(spikes, labels, config.rows)
    S = np.arange(config.rows)
    X, Y = np.meshgrid(S, S)
    ax.plot_surface(X, Y, seq_count.T, edgecolor="grey", lw=0.1, rstride=2, cstride=2,
                alpha=0.4, cmap=cmap)
    
    zticks = (0, 20)
    xyticks = (10, 50, 90)
    ax.contour(X, Y, seq_count.T, zdir="z", offset=-40, cmap=cmap)
    ax.set(
        xlim=(0, config.rows), ylim=(0, config.rows), zlim=(-40, seq_count.max()),
        xticks=xyticks, yticks=xyticks, zticks=zticks,
        xlabel="X", ylabel="Y", zlabel='Seq. count', title="Sequence Landscape")
    
    elev = 20   # defines the angle of the camera location above the x-y plane.
    azim = -105 # rotates the camera about the vertical axis, with a positive angle corresponding to a right-handed rotation. 
    roll = 0    # rotates the camera about the viewing axis.
    ax.view_init(elev, azim, roll)
    
    
    
    
    from skimage.measure import find_contours
    seq_counts = np.zeros((seeds.size, config.rows, config.rows), dtype=int)
    mask = np.zeros((config.rows, config.rows), dtype=bool)
    for s, seed in enumerate(seeds):
        tag = config.baseline_tag(seed=seed)
        spikes, labels = PIC.load_spike_train(tag, config)
        seq_count = _get_sequence_landscape(spikes, labels, config.rows)
        seq_counts[s] = seq_count
        force_forest = False
        fname = f"bridge_{s}"
        try:
            if force_forest:
                raise FileNotFoundError
            bridge_neurons = PIC.load(fname)
        except FileNotFoundError:
            forest, merges, bridge_neurons = grow_forest(seq_count)
            PIC.save(fname, bridge_neurons)
        mask[bridge_neurons[:, 1], bridge_neurons[:, 0]] = True
    
    ax = fig.add_subplot(gs[:, 1])
    im = create_image(seq_counts.mean(axis=0).T, cmap=cmap, axis=ax)
    ax.set(
        xticks=xyticks, yticks=xyticks,
        xlabel="X", ylabel="Y", title="Sequence Counts")
    cbar = add_colorbar_from_im(ax, im)
    cbar.set_ticks(np.linspace(0, 30, 4, dtype=int))
    cbar.set_label("Seq. count", rotation=270, labelpad=10)
    
    contours = find_contours(mask, 0.5)
    contour_kwargs = {"color": "lime", "linewidth": 1}
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], **contour_kwargs)
        
    
    fname = f"bridge_{example_seed}"
    
    # bridge_neurons = PIC.load(fname)
    conn = ConnectivityMatrix(config)
    indegree, _ = conn.degree(conn._EE)
    indegree = indegree * config.synapse.weight

    tag = config.baseline_tag(seed=example_seed)
    avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
    
    ax = fig.add_subplot(gs[1, 3])
    remove_topright_spines(ax)
    ax.set(xlabel="In-degree", ylabel="Avg. rate", yticks=(0, 0.1, 0.2, 0.3), xlim=(indegree_low, indegree_high))
    scatter_kwargs = {"marker": ".", "s": 8, "edgecolor": 'none'}
    ax.scatter(indegree.flatten(), avgRate, **scatter_kwargs)
    ax.scatter(indegree[mask].flatten(), avgRate[mask.flatten()], c=contour_kwargs["color"], **scatter_kwargs)
    
    
    ax_hist = fig.add_subplot(gs[0, 3])
    ax_hist.set(title="Semi-Transmissive\nNeurons", xlim=(indegree_low, indegree_high))
    remove_spines_and_ticks(ax_hist)
    
    
    hist_kwargs = {"bottom": 0.2, "range": (indegree_low, indegree_high), "bins": 15, "rwidth": 0.8}
    H, edges, _ = ax_hist.hist(indegree.flatten(), **hist_kwargs)
    bridge_degrees = indegree[mask].flatten()
    ax_hist.hist(bridge_degrees, **hist_kwargs, color=contour_kwargs["color"])
    
    separator = np.linspace(indegree.min(), indegree.max(), 5+1)
    from figure_generator.figure2 import map_indegree_to_color
    for sep in separator[1:-1]:
        color = map_indegree_to_color(sep, indegree_low, indegree_high)
        ax_hist.axvline(sep, ls="--", c=color, zorder=12)
        ax.axvline(sep, ls="--", c=color, zorder=12)
        
    PIC.save_figure(filename, fig)
#===============================================================================
# METHODS
#===============================================================================



#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
