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
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from plot.constants import *
from lib.neuralhdf5 import NeuralHdf5, default_filename
import lib.universal as UNI
import lib.pickler as PIC
from params import RandomConfig
config = RandomConfig() # Use a specific one here!
#===============================================================================
# CONSTANTS
#===============================================================================

rcParams["font.size"] = 8
rcParams["figure.figsize"] = (18*cm, 10*cm)
rcParams["legend.fontsize"] = 7

title_style = {
    "fontsize": plt.rcParams["axes.titlesize"],
    "fontweight": plt.rcParams["axes.titleweight"],
    "fontfamily": plt.rcParams["font.family"],
    "ha": "center",
    "va": "center"
}

filename = "random_location"
degree_cmap = plt.cm.jet
min_degree = 800
max_degree = 1250
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=5, width_ratios=(1, 1, .35, 1, 1))
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.9,
        wspace=0.15,
        hspace=0.8
    )
    # xtitle = 0.2, 0.7
    # ytitle = 0.9, 0.5
    # fig.text(xtitle[0], ytitle[0], "Random Patches", **title_style)
    # fig.text(xtitle[1], ytitle[0], "Random Neurons", **title_style)
    # fig.text(xtitle[0], ytitle[1], r"$\Delta$ Sequence Count", **title_style)
    # fig.text(xtitle[1], ytitle[1], r"$\Delta$ Avg. Duration", **title_style)
    
    ##### TOP LEFT: RANDOM PATCH LOCATIONS ##############################
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_pos.set_title("Patch: +10%")
    ax_pos.set_xlabel("Sequence count")
    ax_pos.set_ylabel("Avg. duration [ms]")
    yticks = np.arange(160, 225, 15)
    ylim = (152, 210)
    xticks = np.arange(90, 130, 15)
    xlim = (88, 116)
    ax_pos.set_yticks(yticks)
    ax_pos.set_ylim(ylim)
    ax_pos.set_xticks(xticks)
    ax_pos.set_xlim(xlim)
    
    ax_neg = fig.add_subplot(gs[0, 1])
    ax_neg.set_title("Patch: -10%")
    ax_neg.set_xlabel("Sequence count")
    ax_neg.set_xticks(xticks)
    ax_neg.set_xlim(xlim)
    ax_neg.set_yticks(yticks)
    ax_neg.set_ylim(ylim)
    ax_neg.tick_params(labelleft=False)
    
    panel_random_patch_locations(ax_pos, config, p=0.1)
    panel_random_patch_locations(ax_neg, config, p=-0.1)
    ax_neg.legend(handletextpad=0.1)
    ##### TOP RIGHT: RANDOM NEURONS ######################################
    ax_pos = fig.add_subplot(gs[0, -2])
    ax_neg = fig.add_subplot(gs[0, -1])
    # ax_pos.text(xmid, ymax, r"$\Delta$ sequence count")
    ax_pos.set_title("Patch: +10%")
    ax_neg.set_title("Patch: -10%")
    ax_pos.set_xlabel("Sequence count")
    ax_pos.set_ylabel("Avg. duration [ms]")
    ax_pos.set_xticks(xticks)
    ax_pos.set_xlim(xlim)
    ax_pos.set_yticks(yticks)
    ax_pos.set_ylim(ylim)
    
    ax_neg.set_xlabel("Sequence count")
    ax_neg.set_xticks(xticks)
    ax_neg.set_xlim(xlim)
    ax_neg.set_yticks(yticks)
    ax_neg.set_ylim(ylim)
    ax_neg.tick_params(labelleft=False)
    
    tmp_config = copy.copy(config)
    tmp_radius = config.radius[0]
    tmp_config.radius = 80 
    panel_random_patch_locations(ax_pos, tmp_config, p=0.1)
    panel_random_patch_locations(ax_neg, tmp_config, p=-0.1)
    assert config.radius[0] == tmp_radius
    ax_neg.legend(handletextpad=0.1)
    

    ##### BOTTOM LEFT: PATCH SEQUENCES ###################################
    ax_pos = fig.add_subplot(gs[1, 0])
    ax_neg = fig.add_subplot(gs[1, 1])
    ax_pos.set_title("Patch: +10%")
    ax_neg.set_title("Patch: -10%")
    ax_pos.set_xlabel("Mean patch in-degree", labelpad=3)
    ax_pos.set_ylabel(r"$\Delta$ sequence count", labelpad=3)
    x_ticks = (800, 1000, 1200)
    y_ticks = (-20, -10, 0, 10, 20)
    ylim = (-13, 13)
    ax_pos.set_xticks(x_ticks)
    ax_pos.set_yticks(y_ticks)
    ax_pos.set_ylim(ylim)
    
    ax_neg.set_xlabel("Mean patch in-degree")
    ax_neg.set_xticks(x_ticks)
    ax_neg.set_yticks(y_ticks)
    ax_neg.set_ylim(ylim)
    ax_neg.tick_params(labelleft=False)
    
    panel_feature_over_indegree(ax_pos, config, feature="count", p=0.1)
    panel_feature_over_indegree(ax_neg, config, feature="count", p=-0.1)
    
    
    ##### BOTTOM RIGHT: PATCH DURATIONS ##################################
    ax_pos = fig.add_subplot(gs[1, -2])
    ax_neg = fig.add_subplot(gs[1, -1])
    ax_neg.set_ylim(ylim)
    ax_pos.set_title("Patch: +10%")
    ax_neg.set_title("Patch: -10%")
    ax_pos.set_xlabel("Mean patch in-degree")
    ax_pos.set_ylabel("$\Delta$ avg. duration [ms]")
    y_ticks = (-30, -15, 0, 15, 30)
    ylim = (-37, 37)
    ax_pos.set_xticks(x_ticks)
    ax_pos.set_yticks(y_ticks)
    ax_pos.set_ylim(ylim)
    
    ax_neg.set_xlabel("Mean patch in-degree")
    ax_neg.set_xticks(x_ticks)
    ax_neg.set_yticks(y_ticks)
    ax_neg.set_ylim(ylim)
    ax_neg.tick_params(labelleft=False)
    
    panel_feature_over_indegree(ax_pos, config, feature="duration", p=0.1)
    panel_feature_over_indegree(ax_neg, config, feature="duration", p=-0.1)
    
    PIC.save_figure(filename, fig)
#===============================================================================
# METHODS
#===============================================================================
def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")

def panel_random_patch_locations(ax:object, config:object, p:float):
    plot_kwargs = {
        "marker": ".",
        "capsize": 2,
        # "markeredgecolor": "k",
        # "markerfacecolor": "k"
    }
    bs_kwargs = {"label": "baseline", "zorder": 20, "ls": "none", "color": BS_COLOR, "markersize": 10,}
    mean_kwargs = {"label": "mean", "zorder": 20, "ls": "none", "color": "k", "markersize": 10,}
    
    tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
    
    # from matplotlib.colors import TABLEAU_COLORS
    # fig, axint = plt.subplots(num="bs"+tags_by_seed[0][0])
    # fig, axint_p = plt.subplots(num=tags_by_seed[0][0])
    with NeuralHdf5(default_filename, "a", config=config) as file:
        bs_durations = np.zeros(config.drive.seeds.size)
        bs_counts    = np.zeros(config.drive.seeds.size)
        for t, tag in enumerate(config.baseline_tags):
            file.reset_sequence_duration_and_count(tag, is_baseline=True)
            durations, counts = file.get_sequence_duration_and_count(tag, is_baseline=True)
            # axint.hist(durations, bins=np.arange(0, 650, 30), facecolor='none', edgecolor=list(TABLEAU_COLORS.values())[t], lw=3)
            bs_durations[t] = durations.mean()
            bs_counts[t] = counts
        ax.errorbar(bs_counts.mean(), bs_durations.mean(), 
                    xerr=bs_counts.std(ddof=1) / np.sqrt(bs_counts.size),
                    yerr=bs_durations.std(ddof=1) / np.sqrt(bs_durations.size),
                    **plot_kwargs, **bs_kwargs)
        
        patch_count_means = np.zeros(len(tags_by_seed))
        patch_duration_means = np.zeros(len(tags_by_seed))
        for s, tags in enumerate(tags_by_seed):
            patch_durations = np.zeros(config.drive.seeds.size)
            patch_counts    = np.zeros(config.drive.seeds.size)
            for t, tag in enumerate(tags):
                # file.reset_sequence_duration_and_count(tag)
                durations, counts = file.get_sequence_duration_and_count(tag)
                patch_durations[t] = durations.mean()
                # axint_p.hist(durations, bins=np.arange(0, 650, 30), facecolor='none', edgecolor=list(TABLEAU_COLORS.values())[t], lw=3)
                patch_counts[t] = counts
            # Get indegree and color!
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
            radius  = UNI.radius_from_tag(tag)
             
            color = map_indegree_to_color(file.get_indegree(center, radius))
            patch_count_means[s] = patch_counts.mean()
            patch_duration_means[s] = patch_durations.mean()
            ax.errorbar(patch_counts.mean(), patch_durations.mean(), 
                        xerr=patch_counts.std(ddof=1) / np.sqrt(patch_counts.size),
                        yerr=patch_durations.std(ddof=1) / np.sqrt(patch_durations.size),
                        color=color, **plot_kwargs)
        # ax.errorbar(patch_count_means.mean(), patch_duration_means.mean(),
        #             xerr=patch_count_means.std(ddof=1) / np.sqrt(patch_count_means.size),
        #             yerr=patch_duration_means.std(ddof=1) / np.sqrt(patch_duration_means.size),
        #             **plot_kwargs, **mean_kwargs,
        # )
        
            
            
            
def panel_feature_over_indegree(ax:object, config:object, feature:str, p:float):
    linewidth = 2
    ax.axhline(c=BS_COLOR, lw=linewidth, label="baseline")
    plot_kwargs = {
        "marker": "o",
        "capsize": 4,
    }
    with NeuralHdf5(default_filename, "a", config=config) as file:
        # Plot Baseline
        bs_durations = np.zeros(config.drive.seeds.size)
        bs_counts    = np.zeros(config.drive.seeds.size)
        for t, tag in enumerate(config.baseline_tags):
            durations, counts = file.get_sequence_duration_and_count(tag, is_baseline=True)
            bs_durations[t] = durations.mean()
            bs_counts[t] = counts
        if feature == "duration":
            SEM = bs_durations.std(ddof=1) / np.sqrt(bs_durations.size)
        elif feature == "count":
            SEM = bs_counts.std(ddof=1) / np.sqrt(bs_counts.size)
        else:
            raise ValueError("No valid feature selected")
        ax.axhline(SEM, c=BS_COLOR, lw=linewidth, ls="--")
        ax.axhline(-SEM, c=BS_COLOR, lw=linewidth, ls="--")
        ax.axhspan(-SEM, SEM, color=BS_COLOR, alpha=0.075)

        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])  
        for tags in tags_by_seed:
            patch_durations = np.zeros(config.drive.seeds.size)
            patch_counts    = np.zeros(config.drive.seeds.size)
            for t, tag in enumerate(tags):
                durations, counts = file.get_sequence_duration_and_count(tag)
                patch_durations[t] = durations.mean()
                patch_counts[t] = counts
                
            if feature == "duration":
                mean = patch_durations.mean() - bs_durations.mean()
                SEM = patch_durations.std(ddof=1) / np.sqrt(patch_durations.size)
            elif feature == "count":
                mean = patch_counts.mean() - bs_counts.mean()
                SEM = patch_counts.std(ddof=1) / np.sqrt(patch_counts.size)
            else:
                raise ValueError("No valid feature selected")
            
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
            radius  = UNI.radius_from_tag(tag)
            indegree = file.get_indegree(center, radius)
            color = map_indegree_to_color(indegree)
            ax.errorbar(indegree, mean, yerr=SEM, color=color, **plot_kwargs)
        


def map_indegree_to_color(indegree:float, min_degree:float=min_degree, max_degree:float=max_degree) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = degree_cmap((indegree - min_degree) / (max_degree - min_degree))
    return color

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
