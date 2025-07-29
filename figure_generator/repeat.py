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

from collections import OrderedDict
import matplotlib.pyplot as plt

from analysis.sequence_correlation import SequenceCorrelator
from params.config_handler import config
from params.motifconfig import RepeatConfig, FakeRepeatConfig, StartConfig
import lib.universal as UNI
import lib.pickler as PIC

from figure_generator.lib import BarPlotter, bar, reorder



#===============================================================================
# CONSTANTS
#===============================================================================

config = RepeatConfig()
tags = "repeat",
tags = "repeat-main",

config = FakeRepeatConfig()
# tags = "fake-repeat", # better for seed 1
tags = "anti-repeat", "fake-repeat", "con-repeat", # better for seed 0

# config = StartConfig()
# tags = "start",


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    all_tags = config.get_all_tags(tags)
    correlator = SequenceCorrelator(config)

    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=False, force_baseline=False)

    for tag in tags:
        plot_sequence_count(config, tag)


#===============================================================================
# STARTER
#===============================================================================

def plot_sequence_count(config, tag):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag)
    if len(detection_spots) == 2:
        keys = ["0", "not 0", "all"]
        labels = [
            "pre",
            "post",
            r"pre$\,$&$\,$post",
        ]

        order = [
            "pre",
            "post",
            r"pre$\,$&$\,$post",
        ]
    elif len(detection_spots) == 3 and tag != "repeat-alt":
        keys = ["0", "not 0", "1", "not 1", "2", "not 2", "all"]
        # early, pre, post
        labels = [
            "early",
            "pre & post",
            "pre",
            "early & post",
            "post",
            "early & pre",
            "all"
        ]


        order = [
            "early",
            "pre",
            "pre & post",
            "early & pre",
            "all",
            "early & post",
            "post",
        ]
    elif len(detection_spots) == 3 and tag == "repeat-alt":
        # pre, post, right
        labels = [
            "pre",
            "post & right",
            "post",
            "pre & right",
            "right",
            "pre & post",
            "all"
        ]
        order = [
            "pre",
            "post",
            "post & right",
            "pre & post",
            "all",
            "pre & right",
            "right",
        ]

    for tag_cross_seeds in config.get_all_tags(tag, seeds="all"):

        barplotter = BarPlotter(config, tag_cross_seeds, labels, detection_spots)

        name, _ = UNI.split_seed_from_tag(tag_cross_seeds[0])
        ### Figure
        # fig, axes = plt.subplots(ncols=1, sharey=True, num=name, figsize=(3, 4), tight_layout=True)
        # axes = [axes]
        # # fig, axes = plt.subplots(ncols=len(tag_cross_seeds) + 1, sharey=True, num=name)
        # fig.suptitle(name)
        # barplotter.init_axes(axes)

        ### Baseline - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
        shared_all_seeds = reorder(shared_all_seeds, order)

        print("Baseline")
        for key, value in shared_all_seeds.items():
            print(f"{key}:", value, value.mean(), value.std(ddof=1))
        print()
        ### Plotting
        fig = bar(order, shared_all_seeds, name)
        # barplotter.bar_sequences(shared_all_seeds, axes, is_baseline=True)


        ### Patch - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
        shared_all_seeds = reorder(shared_all_seeds, order)
        print("Patch:", tag_cross_seeds[0])
        for key, value in shared_all_seeds.items():
            print(f"{key}:", value, value.mean(), value.std(ddof=1))
        print()
        ### Plotting
        fig = bar(order, shared_all_seeds, name)
        plt.legend(loc="upper left",
                   ncol=1, fancybox=True, shadow=True,
                   )

        # barplotter.bar_sequences(shared_all_seeds, axes)
        # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5),
        #            ncol=1, fancybox=True, shadow=True
        #            )
        plt.xlabel(" ")
        # plt.tight_layout()
        PIC.save_figure(f"{name}_across_seeds_{detection_spots}", fig,
                        sub_directory=config.sub_dir, transparent=True)





if __name__ == '__main__':
    main()
    plt.show()
