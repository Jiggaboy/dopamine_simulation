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

from plot.lib import plot_activity

## Specifiy the Config here
from params import PerlinConfig, StarterConfig, ScaleupConfig

NORM = (-.3, .3)

TAG_DELIMITER = "_"
TAG_NAME_INDEX = 0
TAG_RADIUS_INDEX = 1
TAG_SEED_INDEX = -1

def main():
    cf = PerlinConfig()
    all_tags = cf.get_all_tags()
    activity_difference(cf, all_tags)

    
def activity_difference(config:object, postfixes:list, **kwargs):
    center_range = config.center_range
    ################################################################################################################################################
    import pandas as pd
    avgs = pd.DataFrame()
    names = []


    ################################################################################################################################################
    for tag in postfixes:
        seed = tag.split(TAG_DELIMITER)[TAG_SEED_INDEX]
        baseRate = PIC.load_average_rate(config.baseline_tag(seed), sub_directory=config.sub_dir, config=config)
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)

        figname = f"patch_{tag}_bs_{seed}"
        rate_diff = avgRate - baseRate
        plot_activity(rate_diff, figname=figname, norm=NORM, cmap=plt.cm.seismic, figsize=(8, 6))

        ################################################################################################################################################
        import pandas as pd
        df = pd.DataFrame(rate_diff, columns=[tag])
        avgs = pd.concat([avgs, df], axis=1)
        
        
        ################################################################################################################################################
        plot_patch_from_tag(tag, center_range, config)
        
        # Plot the patch
        #############
        # Make Details of the figure here!
        from figure_generator.connectivity_distribution import set_layout
        set_layout(config.rows, margin=0)
        plt.savefig(UNI.get_fig_filename(figname, format_="svg"), format="svg")
        plt.title(f"Network changes: \nActivation difference: {100 * (avgRate - baseRate).mean():+.2f}%")
    
    # Find unique names
    unique_tags = []
    for tag in postfixes:
        unique_tags.extend(tag.rsplit(TAG_DELIMITER, 1)[:TAG_SEED_INDEX])
    logger.info(f"Unique_tags: {unique_tags}")
    unique_tags = set(unique_tags)
    logger.info(f"Unique_tags: {unique_tags}")
    
    # Plot averaged averages
    for tag in unique_tags:
        tags_of_seeds = config.find_tags(tag)
        avg_diffs = avgs[tags_of_seeds]
        plot_activity(avg_diffs.mean(axis=1).to_numpy(), figname=tag, norm=NORM, cmap=plt.cm.seismic, figsize=(8, 6))
        plot_patch_from_tag(tag, center_range, config)
        set_layout(config.rows, margin=0)
        plt.savefig(UNI.get_fig_filename(figname, format_="svg"), format="svg")
        plt.title(f"Network changes: \nActivation difference: {100 * avg_diffs.to_numpy().mean():+.2f}%")
        

def plot_patch_from_tag(tag:str, center_range:tuple, config:object):
    name = tag.split(TAG_DELIMITER)[TAG_NAME_INDEX]
    center = center_range[name]

    radius = tag.split(TAG_DELIMITER)[TAG_RADIUS_INDEX]
    plot_patch(center, float(radius), width=config.rows)
    
        

def plot_patch(center:tuple, radius:int, width:int)->None:
    # Plot the circle on location
    plot_circle(center, radius=radius)

    # Plot the circle on the other side of the toroid
    center = np.asarray(center)
    for idx, c in enumerate(center):
        if c + radius > width:
            n_center = center.copy()
            n_center[idx] = n_center[idx] - width
            plot_circle(n_center, radius=radius)
    # Plot it also, when both sides are exceeded
    if all(center + radius > width):
        n_center = center.copy() - width
        plot_circle(n_center, radius=radius)


def plot_circle(center, radius):
    circle = mpatches.Circle(center, radius=radius, fc="None", ec="black", linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)



if __name__ == "__main__":
    main()
    plt.show()