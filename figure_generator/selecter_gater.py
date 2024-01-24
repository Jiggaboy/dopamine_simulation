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
from params import SelectConfig, GateConfig
from params.config_handler import ExploreConfig, config
import lib.universal as UNI

from figure_generator.lib import BarPlotter



#===============================================================================
# CONSTANTS
#===============================================================================

selecter_tag = "select-alt"
selecter_tag = "select"
gater_tag = "gate-left"
# gater_tag = "gate"

fig_folder = "_figure_3"


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    # config = SelectConfig()
    # plot_selecter(config)
    # config = ExploreConfig()
    plot_gater(config)


#===============================================================================
# SELECTER
#===============================================================================
def plot_selecter(config):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(selecter_tag)
    tags = config.get_all_tags(selecter_tag, seeds="all")

    # base, left, right
    labels = [
        "base",
        "left & right",
        "left",
        "base & right",
        "right",
        "pre & left",
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
        plt.legend()

        PIC.save_figure(f"seq_bar_{name}", fig, sub_directory=fig_folder)
    plt.show()


#===============================================================================
# GATER
#===============================================================================

def plot_gater(config):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(gater_tag)
    tags = config.get_all_tags(gater_tag, seeds="all")

    # left, right, merged
    labels = [
        "left",
        "right & post",
        "right",
        "left & post",
        "post",
        "left & right",
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
