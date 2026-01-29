#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Plots the Gate and the Select Motif
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


from analysis.sequence_correlation import SequenceCorrelator
from figure_generator.lib import BarPlotter, reorder
from params import config
from plot.lib.frame import create_image
from plot.lib.basic import add_colorbar, plot_patch_from_tag, plot_patch, get_color
from plot.constants import COLOR_MAP_ACTIVITY, NORM_ACTIVITY, COLOR_MAP_DIFFERENCE, cm
from lib.neuralhdf5 import NeuralHdf5, default_filename
from plot.lib import remove_spines_and_ticks, remove_topright_spines
import lib.pickler as PIC

from figure_generator.figure3 import panel_avg_activity
from figure_generator.figure1 import xyticks

#===============================================================================
# CONSTANTS
#===============================================================================
rcParams["font.size"] = 8
rcParams["figure.figsize"] = (17.6*cm, 12*cm)
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


legend_kwargs = {"ncol": 2, "loc": "upper center"}


filename = "intersection"

force_recounting_patch = False
# force_recounting_patch = True

force_recounting_baseline = False
# force_recounting_baseline = True
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=5, width_ratios=(1, 1.2, .35, .75, .75))
    fig.subplots_adjust(
        left=0.0,
        right=0.99,
        bottom=0.12,
        top=0.9,
        wspace=0.07,
        hspace=0.4
    )
    col_avg_activity = 1
    col_STAS_left  = 3
    col_STAS_right = 4
    
    #===============================================================================
    # SELECT
    row = 0
    name, p = "select-left", .1
    labels = (
        r"$M$", r"$B_1\,$&$\,B_2$", r"$B_1$", r"$M\,$&$\,B_2$",
        r"$B_2$", r"$M\,$&$\,B_1$", r"all"
    )
    order = (
        r"$M$", r"$M\,$&$\,B_1$", r"$B_1$", r"$M\,$&$\,B_2$",
        r"$B_2$", r"$B_1\,$&$\,B_2$", r"all"
    )
    ax = fig.add_subplot(gs[row, col_avg_activity])
    ax.set_title(f"Activity Difference\nB2: {p:+.0%}")
    ax.set(xlabel="X", xticks=(30, 60, 90), xlim=(28, 93))
    ax.set(ylabel="Y", yticks=(10, 30, 50), ylim=(0, 65))
    ax, cbar = panel_avg_activity(ax, config, name=name, p=p)
    cbar.set_label(r"$\Delta$ activity", rotation=270, labelpad=6)
    
    
    ax = fig.add_subplot(gs[row, col_STAS_left])
    remove_topright_spines(ax)
    ax.set_title(f"Sequence Counts\nB2: {p:+.0%}")
    ax.tick_params(labelbottom=False)
    ax.set_yticks([0, 20, 40],)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Sequence count", labelpad=1)
    panel_STAS_count_intersection(ax, config, name=name, p=p, labels=labels, order=order)    
    ax.legend(**legend_kwargs)
    # return
    
    ax = fig.add_subplot(gs[row, col_STAS_right])
    remove_topright_spines(ax)
    ax.set_title(f"Sequence Counts\nB2: {-p:+.0%}")
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_yticks([0, 20, 40],)
    ax.set_ylim(0, 50)
    panel_STAS_count_intersection(ax, config, name=name, p=-p, labels=labels, order=order)    
    ax.legend(**legend_kwargs)
    #===============================================================================
    # GATE

    name, p = "gate-left", -.1
    row = 1
    labels = (
        r"$B_1$", r"$M\,$&$\,B_2$", r"$B_2$",
        r"$M\,$&$\,B_1$", r"$M$", r"$B_1\,$&$\,B_2$", r"all"
    )
    order = (
        r"$M$", r"$M\,$&$\,B_1$", r"$B_1$", r"$M\,$&$\,B_2$",
        r"$B_2$", r"$B_1\,$&$\,B_2$", r"all"
    )
    
    ax = fig.add_subplot(gs[row, col_avg_activity])
    ax.set_title(f"B1: {p:+.0%}")
    shifted_ticks = ((10, 30, 49, 50, 70, 90), (60, 80, 99, " ", 20, 40))
    ax.set_yticks(*shifted_ticks)
    ax.set(xlabel="X", xticks=(10, 40, 70), xlim=(8, 73))
    ax.set(ylabel="Y", ylim=(20, 85))
    _, cbar = panel_avg_activity(ax, config, name=name, p=p, roll=(50, 0))
    cbar.set_label(r"$\Delta$ activity", rotation=270, labelpad=6)


    ax = fig.add_subplot(gs[row, col_STAS_left])
    remove_topright_spines(ax)
    ax.set_title(f"B1: {p:+.0%}")
    ax.tick_params(labelbottom=True)
    ax.set_yticks([0, 20, 40],)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Sequence count", labelpad=1)
    panel_STAS_count_intersection(ax, config, name=name, p=p, labels=labels, order=order)    
    ax.legend(**legend_kwargs)

    ax = fig.add_subplot(gs[row, col_STAS_right])
    remove_topright_spines(ax)
    ax.set_title(f"B1: {-p:+.0%}")
    ax.tick_params(labelbottom=True, labelleft=False)
    ax.set_yticks([0, 20, 40],)
    ax.set_ylim(0, 50)
    panel_STAS_count_intersection(ax, config, name=name, p=-p, labels=labels, order=order)    
    ax.legend(**legend_kwargs)
    
    PIC.save_figure(filename, fig, transparent=True)
#===============================================================================
# METHODS
#===============================================================================
def panel_STAS_count_intersection(ax:object, config:object, name:str, p:float, labels:tuple, order:tuple):
    barwidth = 0.4

    keys = ["0", "not 0", "1", "not 1", "2", "not 2", "all"]

    
    ax.set_xticks(np.arange(len(order)), order, rotation=75,)
    
    
    tags = config.get_all_tags(name, weight_change=p)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tags[0])
    
    correlator = SequenceCorrelator(config)
    for tag in tags:
        correlator.count_shared_sequences(tag, force_patch=force_recounting_patch, force_baseline=force_recounting_baseline)

    barplotter = BarPlotter(config, tags, labels, detection_spots)

    shared_all_seeds_bs = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
    shared_all_seeds_bs = reorder(shared_all_seeds_bs, order)

    avg_bs = [s.mean() for s in shared_all_seeds_bs.values()]
    std_bs = [s.std(ddof=1) for s in shared_all_seeds_bs.values()]
    bar_bs = ax.bar(order, avg_bs, yerr=std_bs, 
           width=-barwidth, align="edge", label="baseline")
    
    ### Patch - Count sequences
    shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
    shared_all_seeds = reorder(shared_all_seeds, order)

    avg = [s.mean() for s in shared_all_seeds.values()]
    std = [s.std(ddof=1) for s in shared_all_seeds.values()]
    ax.bar(order, avg, yerr=std, 
           width=barwidth, align="edge", label="patch")
    
    print(name)
    for key, abs, sbs, a, s in zip(shared_all_seeds.keys(), avg_bs, std_bs, avg, std):
        print(f"{key}: {abs}+-{sbs}; {a}+-{s}")
    
    
    
    #
    # print("For Select")
    # print(name, p)
    # M = shared_all_seeds_bs['$M$'].mean()
    # B1 = shared_all_seeds_bs['$B_1$'].mean()
    # MB1 = shared_all_seeds_bs['$M\,$&$\,B_1$'].mean()
    # print(f"BS: P(B1|M) = (B intersect M) / B = {MB1 / M}")
    # B2 = shared_all_seeds_bs['$B_2$'].mean()
    # MB2 = shared_all_seeds_bs['$M\,$&$\,B_2$'].mean()
    # print(f"BS: P(B2|M) = (B intersect M) / B = {MB2 / M}")
    # print()
    # B1 = shared_all_seeds['$B_1$'].mean()
    # MB1 = shared_all_seeds['$M\,$&$\,B_1$'].mean()
    # print(f"P(B1|M) = (B intersect M) / B = {MB1 / M}")
    # B2 = shared_all_seeds['$B_2$'].mean()
    # MB2 = shared_all_seeds['$M\,$&$\,B_2$'].mean()
    # print(f"P(B2|M) = (B intersect M) / B = {MB2 / M}")
    #
    #
    # print("For Gate")
    # print(name, p)
    # B1 = shared_all_seeds_bs['$B_1$'].mean()
    # MB1 = shared_all_seeds_bs['$M\,$&$\,B_1$'].mean()
    # print(f"BS: P(M|B1) = (B intersect M) / B = {MB1 / B1}")
    # B2 = shared_all_seeds_bs['$B_2$'].mean()
    # MB2 = shared_all_seeds_bs['$M\,$&$\,B_2$'].mean()
    # print(f"BS: P(M|B2) = (B intersect M) / B = {MB2 / B2}")
    # print()
    # B1 = shared_all_seeds['$B_1$'].mean()
    # MB1 = shared_all_seeds['$M\,$&$\,B_1$'].mean()
    # print(f"P(M|B1) = (B intersect M) / B = {MB1 / B1}")
    # B2 = shared_all_seeds['$B_2$'].mean()
    # MB2 = shared_all_seeds['$M\,$&$\,B_2$'].mean()
    # print(f"P(M|B2) = (B intersect M) / B = {MB2 / B2}")
#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
