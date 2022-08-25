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
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util.pickler as PIC
import universal as UNI

from plot.lib import plot_activity, create_image, image_slider_2d, image_slider_1d, plot_patch


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
    all_tags = cf.get_all_tags(seeds="all")
    fig, slider = activity_difference(cf, all_tags)
    
    # Slider has to be assigned in the same scope as plt.show()
    fig, slider = baseline_activity_differences_across_seeds(cf, cf.baseline_tags)
    plt.show()
    
    
###### Just for baseline right now (see figname and title)
def baseline_activity_differences_across_seeds(config:object, tags:list):
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


def create_patch_difference_plot(tag:str, data:np.ndarray, config:object):
    full_name, _ = split_seed_from_tag(tag[0])
    figname = f"Average_diff_patch_{full_name}"
    title = f"Differences in patch against baseline simulation"
    slide_label = "Seed"
    fig, axes = plt.subplots(num=figname, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=TITLE_FONT)
    
    method = partial(update_patch_difference, data=data, fig=fig, axis=axes, tag=tag[0], config=config)
    s = image_slider_1d(data, fig, axis=axes, label=slide_label, method=method)
    return s


def update_patch_difference(data:np.ndarray, fig, axis, tag:str, config:object, idx:int):
    create_image(data[idx], axis=axis, norm=NORM, cmap=CMAP)
    axis.set_title(f"Specifier: {tag} (seed: {idx})")
    plot_patch_from_tag(tag, config)

    
def activity_difference(config:object, postfixes:list, **kwargs):
    slider = []
    # Postfixes is now a list of lists with seeds inspecific tags.
    for tag in postfixes:
        # So, we have now a list with only different seeds here        
        pooled_rates = rate_differences_against_baseline(config, tag)
        # Now, we have all the averaged activity across seeds:
        # Make a slider plot for P - BS (seed specific)
        s = create_patch_difference_plot(tag, pooled_rates.T, config=config)
        slider.append(s)
        # Make a plot of P - BS (averaged)
        create_patch_average_difference_plot(tag, pooled_rates, config)
    return slider


def create_patch_average_difference_plot(tag:list, rates:np.ndarray, config:object):
    full_name, _ = split_seed_from_tag(tag[0])
    plot_activity(rates.mean(axis=1), figname=full_name, norm=NORM, cmap=CMAP, figsize=FIGSIZE)
    plot_patch_from_tag(tag[0], config)
    plt.savefig(UNI.get_fig_filename(full_name, format_="svg"), format="svg")
    plt.title(f"Network changes: \nActivation difference: {100 * rates.mean():+.2f}%")

    

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


def rate_differences_against_baseline(config:object, tags:list)->np.ndarray:
    pooled_rates = pd.DataFrame()

    for t in tags:
        _, seed = split_seed_from_tag(t)
        baseRate = PIC.load_average_rate(config.baseline_tag(seed), sub_directory=config.sub_dir, config=config)
        avgRate = PIC.load_average_rate(t, sub_directory=config.sub_dir, config=config)

        rate_diff = avgRate - baseRate
        tmp = pd.DataFrame(rate_diff, columns=[t])
        pooled_rates = pd.concat([pooled_rates, tmp], axis=1)
    return np.asarray(pooled_rates)


def plot_patch_from_tag(tag:str, config:object):
    name = name_from_tag(tag)
    center = config.get_center(name)

    radius =  radius_from_tag(tag)
    plot_patch(center, float(radius), width=config.rows)
    

# This should go to universal?!?!
def split_seed_from_tag(tag:str)->tuple:
    return tag.rsplit(TAG_DELIMITER, 1)


def name_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_NAME_INDEX]


def radius_from_tag(tag:str)->tuple:
    return tag.split(TAG_DELIMITER)[TAG_RADIUS_INDEX]
    
if __name__ == "__main__":
    main()
    plt.show()