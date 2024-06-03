#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


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
from collections import OrderedDict

from lib import pickler as PIC
from params import GateRepeatConfig
import lib.universal as UNI

from figure_generator.lib import BarPlotter



#===============================================================================
# CONSTANTS
#===============================================================================

starter_tag = "starter"
repeater_tag = "repeat-early"

fig_folder = "_figure_2"


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    config = GateRepeatConfig()
    plot_starter(config)
    plot_repeater(config)


#===============================================================================
# REPEATER
#===============================================================================
def plot_repeater(config):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(repeater_tag)
    tags = config.get_all_tags(repeater_tag, seeds="all")

    labels = [
        "pre",
        "post",
        "pre & post",
    ]

    for tag_cross_seeds in tags:
        barplotter = BarPlotter(config, tag_cross_seeds, labels, detection_spots)

        name, _ = UNI.split_seed_from_tag(tag_cross_seeds[0])
        fig, axes = plt.subplots(ncols=len(tag_cross_seeds) + 1, sharey=True, num=name)
        fig.suptitle(name)

        barplotter.init_axes(axes)

        ### Baseline - Count sequences
        keys = ["0", "1", "all"]
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
        ### Plotting
        barplotter.bar_sequences(shared_all_seeds, axes, is_baseline=True)


        ### Patch - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
        ### Plotting
        barplotter.bar_sequences(shared_all_seeds, axes)
        plt.legend()

        PIC.save_figure(f"seq_bar_{name}", fig, sub_directory=fig_folder)
    plt.show()


#===============================================================================
# STARTER
#===============================================================================

def plot_starter(config):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(starter_tag)
    tags = config.get_all_tags(starter_tag, seeds="all")

    labels = [
        "pre",
        "middle & post",
        "middle",
        "pre & post",
        "post",
        "pre & middle",
        "all"
    ]


    for tag_cross_seeds in tags:
        barplotter = BarPlotter(config, tag_cross_seeds, labels, detection_spots)

        name, _ = UNI.split_seed_from_tag(tag_cross_seeds[0])
        fig, axes = plt.subplots(ncols=len(tag_cross_seeds) + 1, sharey=True, num=name)
        fig.suptitle(name)
        barplotter.init_axes(axes)

        ### Baseline - Count sequences
        keys = ["0", "not 0", "1", "not 1", "2", "not 2", "all"]
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
        ### Plotting
        barplotter.bar_sequences(shared_all_seeds, axes, is_baseline=True)


        ### Patch - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
        ### Plotting
        barplotter.bar_sequences(shared_all_seeds, axes)
        plt.legend(#loc="center left", #bbox_to_anchor=(1, 0.5),
                   ncol=1, fancybox=True, shadow=True
                   )
        # plt.tight_layout()


    plt.show()






if __name__ == '__main__':
    main()
