#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:11:21 2022

@author: hauke
"""

import cflogger
logger = cflogger.getLogger()

from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util.pickler as PIC
import universal as UNI

from plot.lib import plot_activity, create_image, image_slider_2d, plot_patch


## Specifiy the Config here
from params import PerlinConfig, StarterConfig, ScaleupConfig

NORM = (-.3, .3)
CMAP = plt.cm.seismic
FIGSIZE = (8, 6)
TITLE_FONT = 20

TAG_DELIMITER = "_"
TAG_NAME_INDEX = 0
TAG_RADIUS_INDEX = 1
TAG_SEED_INDEX = -1

def main():
    cf = PerlinConfig()
    all_tags = cf.get_all_tags()
    activity_difference(cf, all_tags)
    
    # Slider has to be assigned in the same scope as plt.show()
    #fig, slider = activity_differences_across_seeds(cf, cf.baseline_tags)
    plt.show()
    
    
###### Just for baseline right now (see figname and title)
def activity_differences_across_seeds(config:object, tags:list):
    """
    Calculates the avg. differences between all simulations given by tags.
    Create a plot with sliders to see the differences.
    """
    figname = "Differences in baseline conditions"
    title = "Differences in baseline conditions"
    
    pooled_diffs = rate_differences(config, tags)
    fig, axes = plt.subplots(num=figname, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=TITLE_FONT)
    slider = image_slider_2d(pooled_diffs, fig, axis=axes, label="seed", norm=NORM, cmap=CMAP)
    return fig, slider
    

def rate_differences(config:object, tags:list)->np.ndarray:
    """
    Calculates the avg. differences between all simulations given by tags.
    """
    tags = UNI.make_iterable(tags)
    pooled_diffs = np.zeros((len(tags), len(tags), config.no_exc_neurons), dtype=float)
    
    for i, tag1 in enumerate(tags):
        avg_rate1 = PIC.load_average_rate(tag1, sub_directory=config.sub_dir, config=config)
        for j, tag2 in enumerate(tags):
            avg_rate2 = PIC.load_average_rate(tag2, sub_directory=config.sub_dir, config=config)
            rate_diff = avg_rate1 - avg_rate2
            pooled_diffs[i, j] = rate_diff
    return np.asarray(pooled_diffs)
    
"""
def _rate_differences(config:object, tags:list):
    #np.triu_indices(4, k=1)
    pooled_diffs = []
    
    for i, tag1 in enumerate(tags):
        avg_rate1 = PIC.load_average_rate(tag1, sub_directory=config.sub_dir, config=config)
        for j, tag2 in enumerate(tags):
            # Skip identical simulations
            if i >= j:
                continue
            avg_rate2 = PIC.load_average_rate(tag2, sub_directory=config.sub_dir, config=config)
    
            rate_diff = avg_rate1 - avg_rate2
            pooled_diffs.append(rate_diff)
    return np.asarray(pooled_diffs)
"""

    
    
def activity_difference(config:object, postfixes:list, **kwargs):
    center_range = config.center_range
    ################################################################################################################################################
    import pandas as pd
    avgs = pd.DataFrame()


    ################################################################################################################################################
    # Here the usual difference between a patch and the baseline is calculated and plotted.
    for tag in postfixes:
        seed = tag.split(TAG_DELIMITER)[TAG_SEED_INDEX]
        baseRate = PIC.load_average_rate(config.baseline_tag(seed), sub_directory=config.sub_dir, config=config)
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)

        figname = f"patch_{tag}_bs_{seed}"
        rate_diff = avgRate - baseRate
        plot_activity(rate_diff, figname=figname, norm=NORM, cmap=CMAP, figsize=FIGSIZE)
        
        # Details of the plot
        # Plot the patch
        plot_patch_from_tag(tag, center_range, config)
        # Make Details of the figure here!
        plt.savefig(UNI.get_fig_filename(figname, format_="svg"), format="svg")
        plt.title(f"Network changes: \nActivation difference: {100 * (avgRate - baseRate).mean():+.2f}%")
        
        
        
        
        ################################################################################################################################################
        # Assign the current rate difference to a column with the tag as a name
        df = pd.DataFrame(rate_diff, columns=[tag])
        avgs = pd.concat([avgs, df], axis=1)
        ################################################################################################################################################
    
    # Find unique names (independent of the seed)
    unique_tags = []
    for tag in postfixes:
        unique_tags.extend(tag.rsplit(TAG_DELIMITER, 1)[:TAG_SEED_INDEX])
    logger.info(f"Unique_tags: {unique_tags}")
    unique_tags = set(unique_tags)
    logger.info(f"Unique_tags: {unique_tags}")
    
    # Plot averaged averages
    for tag in unique_tags:
        # Find all tags which only have a different seed.
        tags_of_seeds = config.find_tags(tag)
        avg_diffs = avgs[tags_of_seeds]
        plot_activity(avg_diffs.mean(axis=1).to_numpy(), figname=tag, norm=NORM, cmap=CMAP, figsize=FIGSIZE)
        plot_patch_from_tag(tag, center_range, config)
        plt.savefig(UNI.get_fig_filename(figname, format_="svg"), format="svg")
        plt.title(f"Network changes: \nActivation difference: {100 * avg_diffs.to_numpy().mean():+.2f}%")




def plot_patch_from_tag(tag:str, center_range:tuple, config:object):
    name = tag.split(TAG_DELIMITER)[TAG_NAME_INDEX]
    center = center_range[name]

    radius = tag.split(TAG_DELIMITER)[TAG_RADIUS_INDEX]
    plot_patch(center, float(radius), width=config.rows)
    

if __name__ == "__main__":
    main()
    plt.show()