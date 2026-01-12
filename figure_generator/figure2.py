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
from params import LocationConfig
config = LocationConfig() # Use a specific one here!
#===============================================================================
# CONSTANTS
#===============================================================================

rcParams["font.size"] = 12
rcParams["figure.figsize"] = (18*cm, 8*cm)

degree_cmap = plt.cm.jet
min_degree = 575
max_degree = 850
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure(layout="constrained")
    subfigs = fig.subfigures(nrows=2, ncols=2)
    axd = {}
    
    ##### TOP LEFT: RANDOM PATCH LOCATIONS ##############################
    axd["upper left"] = subfigs[0, 0].subplots(ncols=2, sharey=True)
    # Panel for random patch locations
    ax_pos, ax_neg = axd["upper left"]
    ax_pos.set_xlabel("Sequence count")
    ax_pos.set_ylabel("Avg. duration [ms]")
    x_ticks = (85, 95, 105)
    xlim = (78, 111)
    yticks = (230, 290, 350)
    ylim = (217, 363)
    ax_pos.set_yticks(yticks)
    ax_pos.set_ylim(ylim)
    ax_pos.set_xticks(x_ticks)
    ax_pos.set_xlim(xlim)
    
    ax_neg.set_xlabel("Sequence count")
    ax_neg.set_xticks(x_ticks)
    ax_neg.set_xlim(xlim)
    
    panel_random_patch_locations(ax_pos, config, p=0.2)
    panel_random_patch_locations(ax_neg, config, p=-0.2)
    ax_neg.legend(handletextpad=0.1)
    
    ##### TOP RIGHT: RANDOM NEURONS ######################################
    axd["upper right"] = subfigs[0, 1].subplots(ncols=2, sharey=True)
    ax_pos, ax_neg = axd["upper right"]
    ax_pos.set_xlabel("Sequence count")
    ax_pos.set_ylabel("Avg. duration [ms]")
    ax_pos.set_xticks(x_ticks)
    ax_pos.set_xlim(xlim)
    ax_pos.set_yticks(yticks)
    ax_pos.set_ylim(ylim)
    
    ax_neg.set_xlabel("Sequence count")
    ax_neg.set_xticks(x_ticks)
    ax_neg.set_xlim(xlim)
    
    tmp_config = copy.copy(config)
    tmp_radius = config.radius[0]
    tmp_config.radius = config.rows
    # panel_random_patch_locations(ax_pos, tmp_config, p=0.2)
    # panel_random_patch_locations(ax_neg, tmp_config, p=-0.2)
    print(config.radius)
    # ax_neg.legend(handletextpad=0.1)
    

    ##### BOTTOM LEFT: PATCH SEQUENCES ###################################
    axd["bottom left"] = subfigs[1, 0].subplots(ncols=2, sharey=True)
    ax_pos, ax_neg = axd["bottom left"]
    ax_pos.set_xlabel("Mean Patch Indegree [au]")
    ax_pos.set_ylabel("Difference in sequence count")
    x_ticks = (600, 700, 800)
    y_ticks = (-20, -10, 0, 10, 20)
    ax_pos.set_xticks(x_ticks)
    ax_pos.set_yticks(y_ticks)
    
    ax_neg.set_ylabel("Difference in sequence count")
    ax_neg.set_xticks(x_ticks)
    
    panel_feature_over_indegree(ax_pos, config, feature="count", p=0.2)
    panel_feature_over_indegree(ax_neg, config, feature="count", p=-0.2)
    
    
    ##### BOTTOM RIGHT: PATCH DURATIONS ##################################
    axd["bottom right"] = subfigs[1, 1].subplots(ncols=2, sharey=True)
    ax_pos, ax_neg = axd["bottom right"]
    ax_pos.set_xlabel("Mean Patch Indegree [au]")
    ax_pos.set_ylabel("Avg. duration [ms]")
    y_ticks = (-50, 0, 50)
    ax_pos.set_xticks(x_ticks)
    ax_pos.set_yticks(y_ticks)
    
    ax_neg.set_xlabel("Mean Patch Indegree [au]")
    ax_neg.set_xticks(x_ticks)
    
    panel_feature_over_indegree(ax_pos, config, feature="duration", p=0.2)
    panel_feature_over_indegree(ax_neg, config, feature="duration", p=-0.2)
    
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
    bs_kwargs = {"label": "baseline", "zorder": 20, "ls": "none", "color": BS_COLOR}
    
    tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
    
    with NeuralHdf5(default_filename, "a", config=config) as file:
        bs_durations = np.zeros(config.drive.seeds.size)
        bs_counts    = np.zeros(config.drive.seeds.size)
        for t, tag in enumerate(config.baseline_tags):
            durations, counts = file.get_sequence_duration_and_count(tag, is_baseline=True)
            bs_durations[t] = durations.mean()
            bs_counts[t] = counts
        ax.errorbar(bs_counts.mean(), bs_durations.mean(), 
                    xerr=bs_counts.std(ddof=1) / np.sqrt(bs_counts.size),
                    yerr=bs_durations.std(ddof=1) / np.sqrt(bs_durations.size),
                    **plot_kwargs, **bs_kwargs)
        
        for tags in tags_by_seed:
            patch_durations = np.zeros(config.drive.seeds.size)
            patch_counts    = np.zeros(config.drive.seeds.size)
            for t, tag in enumerate(tags):
                durations, counts = file.get_sequence_duration_and_count(tag)
                patch_durations[t] = durations.mean()
                patch_counts[t] = counts
            # Get indegree and color!
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
             
            color = map_indegree_to_color(file.get_indegree(center))
            ax.errorbar(patch_counts.mean(), patch_durations.mean(), 
                        xerr=patch_counts.std(ddof=1) / np.sqrt(patch_counts.size),
                        yerr=patch_durations.std(ddof=1) / np.sqrt(patch_durations.size),
                        color=color, **plot_kwargs)
            
            
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
            indegree = file.get_indegree(center)
            color = map_indegree_to_color(indegree)
            ax.errorbar(indegree, mean, yerr=SEM, color=color, **plot_kwargs)
        


def map_indegree_to_color(indegree:float) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = degree_cmap((indegree - min_degree) / (max_degree - min_degree))
    return color

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
