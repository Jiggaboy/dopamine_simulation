#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Plots the Start, Repeat, Stop, and Fake Repeat Motif
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

from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from params import config, SelectConfig
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag, plot_patch, get_color
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE, cm
from lib.neuralhdf5 import NeuralHdf5, default_filename
from plot.lib import remove_topright_spines
import lib.pickler as PIC
import lib.universal as UNI
from figure_generator.figure1 import xyticks

config = SelectConfig()

#===============================================================================
# CONSTANTS
#===============================================================================
rcParams["font.size"] = 8
rcParams["figure.figsize"] = (17.6*cm, 17.6*cm)
rcParams["legend.fontsize"] = 7
rcParams["legend.markerscale"] = 0.6
rcParams["legend.handlelength"] = 1.25
rcParams["legend.columnspacing"] = 1
rcParams["legend.handletextpad"] = 1
rcParams["legend.labelspacing"] = .1
rcParams["legend.borderpad"] = .25
rcParams["legend.handletextpad"] = .5
rcParams["axes.labelpad"] = 2

legend_kwargs = {"loc": "upper center", "ncol": 2}
cbar_kwargs = {"rotation": 270, "labelpad": 10}

filename = "pathway"
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=4, ncols=3, width_ratios=(.8, 1.8, 1))
    fig.subplots_adjust(
        left=0.0,
        right=0.95,
        bottom=0.1,
        top=0.95,
        wspace=0.2,
        hspace=0.3
    )
    
    
    row = 0
    name, p = "start-1", .1
    
    ax = fig.add_subplot(gs[row, 1])
    ax.set_title("Activity Difference")
    ax.set_ylabel("Y")
    ax.set_xticks(xyticks)
    ax.tick_params(labelbottom=False)
    ax.set_yticks(xyticks)
    _, cbar = panel_avg_activity(ax, config, name=name, p=p)
    cbar.set_label(r"$\Delta$ activity", **cbar_kwargs)

    ax = fig.add_subplot(gs[row, -1])
    remove_topright_spines(ax)
    panel_STAS_count(ax, config, name=name, p=p)
    ax.set_title(f"Sequence Counts\nPatch: {p:+.0%}")
    ax.tick_params(labelbottom=False)
    ax.set_yticks([0, 5, 10, 15],)
    ax.set_ylim(0, 14)
    ax.legend(**legend_kwargs)
    
    name, p = "repeat-2", .1
    ax = fig.add_subplot(gs[1, 1])
    ax.set_ylabel("Y")
    ax.set_xticks(xyticks)
    ax.tick_params(labelbottom=False)
    ax.set_yticks(xyticks)
    _, cbar = panel_avg_activity(ax, config, name=name, p=p)
    cbar.set_label(r"$\Delta$ activity", **cbar_kwargs)
    
    ax = fig.add_subplot(gs[1, -1])
    remove_topright_spines(ax)
    panel_STAS_count(ax, config, name=name, p=p)
    ax.set_title(f"Patch: {p:+.0%}")
    ax.tick_params(labelbottom=False)
    ax.set_yticks([0, 10, 20, 30, 40],)
    ax.set_ylim(0, 46)
    ax.legend(**legend_kwargs)
    
    p = -.1
    ax = fig.add_subplot(gs[2, 1])
    ax.set_ylabel("Y")
    ax.set_xticks(xyticks)
    ax.tick_params(labelbottom=False)
    ax.set_yticks(xyticks)
    _, cbar = panel_avg_activity(ax, config, name=name, p=p)
    cbar.set_label(r"$\Delta$ activity", **cbar_kwargs)

    ax = fig.add_subplot(gs[2, -1])
    remove_topright_spines(ax)
    panel_STAS_count(ax, config, name=name, p=p)
    ax.set_title(f"Patch: {p:+.0%}")
    ax.tick_params(labelbottom=False)
    ax.set_yticks([0, 10, 20, 30, 40],)
    ax.set_ylim(0, 46)
    ax.legend(**legend_kwargs)
    
    name, p = "fake-repeat-2", .1
    ax = fig.add_subplot(gs[3, 1])
    _, cbar = panel_avg_activity(ax, config, name=name, p=p)
    cbar.set_label(r"$\Delta$ activity", **cbar_kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks(xyticks)
    ax.set_yticks(xyticks)
    
    ax = fig.add_subplot(gs[3, -1])
    remove_topright_spines(ax)
    panel_STAS_count(ax, config, name=name, p=p)
    ax.set_title(f"Patch: {p:+.0%}")
    ax.set_yticks([0, 10, 20, 30, 40],)
    ax.set_ylim(0, 46)
    ax.legend(**legend_kwargs)
    
    PIC.save_figure(filename, fig, transparent=True)
#===============================================================================
# METHODS
#===============================================================================
def panel_avg_activity(ax:object, config:object, name:str, p:float, roll:tuple=None):
    norm = (-.15, 0.15)
    cmap = COLOR_MAP_DIFFERENCE
    radius = 6
    
    tags = config.get_all_tags(patchnames=name, radius=radius, weight_change=p)
    bs_tags = config.baseline_tags
    rate_diffs = np.zeros((len(tags), config.no_exc_neurons))
    with NeuralHdf5(default_filename, "a", config=config) as file:
        assert len(tags) == len(bs_tags)
        for t, (tag, bs_tag) in enumerate(zip(tags, bs_tags)):
            rate = file.get_average_rate(tag)
            bs_rate = file.get_average_rate(bs_tag, is_baseline=True)
            rate_diffs[t] = rate - bs_rate
        if rate_diffs.ndim > 1:
            rate_diffs = rate_diffs.mean(axis=0)
        
    rate_diffs_tmp = rate_diffs.reshape((config.rows, config.rows))
    if roll is not None:
        rate_diffs_tmp = np.roll(rate_diffs_tmp, roll, axis=(0, 1))

    create_image(rate_diffs_tmp, norm, cmap, axis=ax)
    cbar = add_colorbar(ax, norm, cmap)
    cbar.set_ticks([-.1, 0., 0.1])
    
    # No patch in config if not plotted
    ec, alpha = get_color(p)
    kwargs = {"ec": ec, "alpha": alpha, "ls": "solid"}
    roll_offset = np.asarray(roll)[::-1] if roll is not None else 0
    center = config.center_range[name]
    center = np.asarray(center) + roll_offset
    plot_patch(center, radius, width=config.rows, axis=ax, **kwargs)
    for ds in config.analysis.dbscan_controls.detection_spots:
        ds_name, spots = ds
        if ds_name == name:
            for spot in spots:
                spot = np.asarray(spot)
                plot_patch(spot + roll_offset, radius=2., width=config.rows, axis=ax, lw=1)
    return ax, cbar


def panel_STAS_count(ax:object, config:object, name:str, p:float):
    barwidth = 0.4
    
    # Copied from repeat
    keys = ["0", "not 0", "all"]
    labels = ["pre", "post", r"pre$\,$&$\,$post",]
    order = ["pre", "post", r"pre$\,$&$\,$post",]
    
    ax.set_xticks(np.arange(len(order)), order, rotation=45,)
    
    ax.set_ylabel("Sequence count")
    
    
    from analysis.sequence_correlation import SequenceCorrelator
    from figure_generator.lib import BarPlotter, reorder
    
    tags = config.get_all_tags(name, weight_change=p)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tags[0])
    
    correlator = SequenceCorrelator(config)
    for tag in tags:
        correlator.count_shared_sequences(tag, force_patch=False, force_baseline=False)

    barplotter = BarPlotter(config, tags, labels, detection_spots)
    
    shared_all_seeds = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
    shared_all_seeds = reorder(shared_all_seeds, order)
    
    avg = [s.mean() for s in shared_all_seeds.values()]
    std = [s.std(ddof=1) for s in shared_all_seeds.values()]
    bar_bs = ax.bar(order, avg, yerr=std, 
           width=-barwidth, align="edge", label="baseline")

    shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
    shared_all_seeds = reorder(shared_all_seeds, order)

    avg = [s.mean() for s in shared_all_seeds.values()]
    std = [s.std(ddof=1) for s in shared_all_seeds.values()]
    ax.bar(order, avg, yerr=std, 
           width=barwidth, align="edge", label="patch")
    

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
