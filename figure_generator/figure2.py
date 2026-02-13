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
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import gaussian_kde

from figure_generator.figure1 import xyticks, panel_indegree, indegree_ticks

from plot.lib import plot_patch, add_colorbar_from_im
from plot.constants import *
from plot.constants import min_degree, max_degree
from lib.neuralhdf5 import NeuralHdf5, default_filename
import lib.universal as UNI
import lib.pickler as PIC
from params import RandomConfig
config = RandomConfig() # Use a specific one here!
#===============================================================================
# CONSTANTS
#===============================================================================

figsize = (17.6*cm, 10*cm)

filename = "random_location"

ylim_duration = (154, 204)
xlim_seq = (91, 112)

yticks_duration = np.arange(160, 225, 20)
xticks_seq = np.arange(95, 120, 10)
#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=5, width_ratios=(1, 1, .5, 1, 1))
    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        bottom=0.12,
        top=0.9,
        wspace=0.15,
        hspace=.9,
    )
    
    ##### TOP LEFT: RANDOM PATCH LOCATIONS ##############################
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_pos.set(title="Patch: +10%", xlabel="Sequence count", ylabel="Avg. duration [ms]",
               xticks=xticks_seq, yticks=yticks_duration, xlim=xlim_seq, ylim=ylim_duration)
    
    ax_neg = fig.add_subplot(gs[0, 1])
    ax_neg.set(title="Patch: -10%", xlabel="Sequence count", #ylabel="Avg. duration [ms]",
               xticks=xticks_seq, yticks=yticks_duration, xlim=xlim_seq, ylim=ylim_duration)
    ax_neg.tick_params(labelleft=False)
    
    panel_random_patch_locations(ax_pos, config, p=0.1)
    panel_random_patch_locations(ax_neg, config, p=-0.1)
    
    tmp_radius = config.radius
    config.radius = 80,
    add_density_contourf(ax_pos, config, p=0.1)
    add_density_contourf(ax_neg, config, p=-0.1)
    config.radius = tmp_radius
    assert config.radius == tmp_radius
    
    ax_neg.legend()
    ##### TOP CENTER: In-Degree with locations ######################################
    subgs = gs[0, -3:-1].subgridspec(1, 3, width_ratios=[0.2, 1, 0.4])
    ax = fig.add_subplot(subgs[0, 1])
    ax.set(title="Patch Locations", xlabel="X", ylabel="Y",
               xticks=xyticks, yticks=xyticks)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    
    im = panel_indegree(ax, config)
    cbar = add_colorbar_from_im(ax, im)
    cbar.set_ticks(indegree_ticks)
    cbar.set_label("In-degree", rotation=270, labelpad=8)
    
    # Add locations
    radius = config.radius[0]
    with NeuralHdf5(default_filename, "a", config=config) as file:
        for center in config.center_range.values():
            ec = map_indegree_to_color(file.get_indegree(center, radius))
            plot_patch(center, radius, width=config.rows, axis=ax, ec=ec)
    
    
    ##### TOP RIGHT: MODULATION STRENGTH ######################################
    subgs = gs[0, -1].subgridspec(1, 2, width_ratios=[0.2, 1])
    ax = fig.add_subplot(subgs[0, 1])
    ax.set(xlabel="Modulation strength", ylabel=r"‖($\Delta \mathrm{N} / \mathrm{N_0}$, $\Delta \mathrm{T} / \mathrm{T_0}$ )‖", title="Effective Modulation",
           yticks=(0., 0.1, .2), ylim=(0, .21))
    
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1., decimals=0))
    # Idea: Plot the effect according to -20, -10, 0, +10, +20%
    panel_modulation(ax, config)
    ##### TOP RIGHT: RANDOM NEURONS ######################################
    # ax_pos = fig.add_subplot(gs[0, -2])
    # ax_pos.set(title="Patch: +10%", xlabel="Sequence count", ylabel="Avg. duration [ms]",
    #            xticks=xticks, yticks=yticks, xlim=xlim_seq, ylim=ylim_duration)
    #
    # ax_neg = fig.add_subplot(gs[0, -1])
    # ax_neg.set(title="Patch: -10%", xlabel="Sequence count", #ylabel="Avg. duration [ms]",
    #            xticks=xticks, yticks=yticks, xlim=xlim_seq, ylim=ylim_duration)
    # ax_neg.tick_params(labelleft=False)
    #
    # tmp_config = copy.copy(config)
    # tmp_radius = config.radius[0]
    # tmp_config.radius = 80 
    # panel_random_patch_locations(ax_pos, tmp_config, p=0.1)
    # panel_random_patch_locations(ax_neg, tmp_config, p=-0.1)
    # assert config.radius[0] == tmp_radius
    # ax_neg.legend()
    # return 
    

    ##### BOTTOM LEFT: PATCH SEQUENCES ###################################
    ax_pos = fig.add_subplot(gs[1, 0])
    ax_neg = fig.add_subplot(gs[1, 1])
    ax_pos.set_title("Patch: +10%")
    ax_neg.set_title("Patch: -10%")
    ax_pos.set_xlabel("Mean patch in-degree", labelpad=3)
    ax_pos.set_ylabel(r"$\Delta$ sequence count", labelpad=3)
    x_ticks = (800, 1000, 1200)
    y_ticks = (-20, -10, 0, 10, 20)
    ylim = (-14, 14)
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
    ylim = (-28, 28)
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
    
    PIC.save_figure(filename, fig, transparent=True)
#===============================================================================
# METHODS
#===============================================================================
def panel_random_patch_locations(ax:object, config:object, p:float):
    plot_kwargs = {
        "marker": ".",
    }
    bs_kwargs = {"label": "baseline", "zorder": 20, "ls": "none", "color": BS_COLOR, "markersize": 8,}
    
    tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
    
    with NeuralHdf5(default_filename, "a", config=config) as file:
        bs_durations = np.zeros(config.drive.seeds.size)
        bs_counts    = np.zeros(config.drive.seeds.size)
        for t, tag in enumerate(config.baseline_tags):
            # file.reset_sequence_duration_and_count(tag, is_baseline=True)
            durations, counts = file.get_sequence_duration_and_count(tag, is_baseline=True)
            bs_durations[t] = durations.mean()
            bs_counts[t] = counts
        ax.errorbar(bs_counts.mean(), bs_durations.mean(), 
                    xerr=bs_counts.std(ddof=1) / np.sqrt(bs_counts.size),
                    yerr=bs_durations.std(ddof=1) / np.sqrt(bs_durations.size),
                    **plot_kwargs, **bs_kwargs)
        
        
        patch_count_means = np.zeros((len(tags_by_seed), config.drive.seeds.size))
        patch_duration_means = np.zeros((len(tags_by_seed), config.drive.seeds.size))
        for s, tags in enumerate(tags_by_seed):
            patch_durations = np.zeros(config.drive.seeds.size)
            patch_counts    = np.zeros(config.drive.seeds.size)
            for t, tag in enumerate(tags):
                # file.reset_sequence_duration_and_count(tag)
                durations, counts = file.get_sequence_duration_and_count(tag)
                patch_durations[t] = durations.mean()
                patch_counts[t] = counts
            # Get indegree and color!
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
            radius  = UNI.radius_from_tag(tag)
             
            color = map_indegree_to_color(file.get_indegree(center, radius))
            patch_count_means[s] = patch_counts
            patch_duration_means[s] = patch_durations
            # plot_kwargs["elinewidth"] = 2.25
            # plot_kwargs["capsize"] = 2
            # plot_kwargs["markeredgewidth"] = 1.4
            # ax.errorbar(patch_counts.mean(), patch_durations.mean(), 
            #             xerr=patch_counts.std(ddof=1) / np.sqrt(patch_counts.size),
            #             yerr=patch_durations.std(ddof=1) / np.sqrt(patch_durations.size),
            #             color="black", **plot_kwargs)
            # plot_kwargs["elinewidth"] = 1.5
            # plot_kwargs["capsize"] = 2.2
            # plot_kwargs["markeredgewidth"] = 1
            ax.errorbar(patch_counts.mean(), patch_durations.mean(), 
                        xerr=patch_counts.std(ddof=1) / np.sqrt(patch_counts.size),
                        yerr=patch_durations.std(ddof=1) / np.sqrt(patch_durations.size),
                        color=color, **plot_kwargs)
        
        
def add_density_contourf(ax:object, config:object, p:float) -> None:
    levels = np.linspace(.9e-3, 1.65e-3, 6)
    bw_method = 0.6
    tiles = 100
    cmap = "Reds" if int(config.radius[0]) < 50 else "Greys"
    
    
    tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])
    with NeuralHdf5(default_filename, "a", config=config) as file:        
        sequence_count_means = np.zeros((len(tags_by_seed), config.drive.seeds.size))
        sequence_duration_means = np.zeros((len(tags_by_seed), config.drive.seeds.size))
        for s, tags in enumerate(tags_by_seed):
            for t, tag in enumerate(tags):
                # file.reset_sequence_duration_and_count(tag)
                durations, counts = file.get_sequence_duration_and_count(tag)
                sequence_count_means[s, t] = counts
                sequence_duration_means[s, t] = durations.mean()

        x = sequence_count_means.flatten()
        y = sequence_duration_means.flatten()
        values = np.vstack([x, y])
        kde = gaussian_kde(values, bw_method=bw_method)
        
        xx, yy = np.meshgrid(
            np.linspace(*xlim_seq, tiles),
            np.linspace(*ylim_duration, tiles)
        )
        
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        ax.contourf(xx, yy, density, levels=levels, cmap=cmap)
            

def panel_modulation(ax:object, config:object):
    percentages = (-.2, -.1, 0, .1, .2)
    subset = {key: value for key, value in config.center_range.items() if key in ("start-1", "loc-0", "low-1", "loc-2", "gate-right", "fake-repeat-2")}
    # subset = {key: value for key, value in config.center_range.items() if key not in ("start-1", "loc-0", "low-1", "loc-2", "gate-right", "repeat-2")}

    with NeuralHdf5(default_filename, "a", config=config) as file:
        for name, center in subset.items():
            change = np.zeros((len(percentages), config.drive.seeds.size))
            for p, percentage in enumerate(percentages):
                if np.isclose(percentage, 0):
                    continue
                tags = config.get_all_tags(name, weight_change=percentage)
            
                for t, tag in enumerate(tags):
                    durations, counts = file.get_sequence_duration_and_count(tag)
                    bs_tag = config.get_baseline_tag_from_tag(tag)
                    durations_bs, counts_bs = file.get_sequence_duration_and_count(bs_tag, is_baseline=True)
                    
                    duration_diff = ((durations.mean() - durations_bs.mean()) / durations_bs.mean())
                    sequence_diff = ((counts.astype(int) - counts_bs.astype(int)) / counts_bs)
                    magnitude = np.linalg.norm([duration_diff, sequence_diff])
    
                    change[p, t] = magnitude
            jitter = np.random.uniform(-0.01, 0.01, size=len(percentages))
            jitter[2] = 0
            color = map_indegree_to_color(file.get_indegree(center, radius=config.radius[0]))
            plot_kwargs = {
                "marker": ".",
                # "markeredgecolor": "k",
                # "markeredgewidth": .6, 
            }
            # plot_kwargs["lw"] = 2.
            # ax.plot(percentages + jitter, change.mean(axis=1), 
            #     **plot_kwargs, color="black")  
            plot_kwargs["lw"] = 1.2
            ax.plot(percentages + jitter, change.mean(axis=1), 
                **plot_kwargs, color=color)            
            
            
def panel_feature_over_indegree(ax:object, config:object, feature:str, p:float):
    linewidth = 2
    ax.axhline(c=BS_COLOR, lw=linewidth, label="baseline")
    plot_kwargs = {
        "marker": ".",
        "markersize": 5,
        "capsize": 2,
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

        indegree = file.get_indegree() * config.synapse.weight
        percentile_1 = int(config.no_exc_neurons * 0.025)
        low = np.sort(indegree.flatten())[percentile_1]
        high = np.sort(indegree.flatten())[-percentile_1]
    
        
        separator = np.linspace(low, high, 5+1)
        for sep in separator[1:-1]:
            color = map_indegree_to_color(sep)
            ax.axvline(sep, ymax=1, ls="--", c=color)
        




#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()
