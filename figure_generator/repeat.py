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
from params.motifconfig import RepeatConfig, FakeRepeatConfig
import lib.universal as UNI

from figure_generator.lib import BarPlotter



#===============================================================================
# CONSTANTS
#===============================================================================

config = RepeatConfig()
repeat_tags = "repeat",
# repeat_tags = "repeat-main",

config = FakeRepeatConfig()
repeat_tags = "fake-repeat",



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    all_tags = config.get_all_tags(repeat_tags)
    correlator = SequenceCorrelator(config)

    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=False, force_baseline=False)

    for tag in repeat_tags:
        plot_repeater(config, tag)


#===============================================================================
# STARTER
#===============================================================================

def plot_repeater(config, tag):
    from lib.pickler_class import Pickler
    pickler = Pickler(config)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag)
    tags = config.get_all_tags(tag, seeds="all")

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
    # Labels for repeat-alt
    # # early, post, right
    # labels = [
    #     "pre",
    #     "post & right",
    #     "post",
    #     "pre & right",
    #     "right",
    #     "pre & post",
    #     "all"
    # ]
    # order = [
    #     "pre",
    #     "post",
    #     "post & right",
    #     "pre & post",
    #     "all",
    #     "pre & right",
    #     "right",
    # ]

    for tag_cross_seeds in tags:
        barplotter = BarPlotter(config, tag_cross_seeds, labels, detection_spots)

        name, _ = UNI.split_seed_from_tag(tag_cross_seeds[0])
        ### Figure
        fig, axes = plt.subplots(ncols=1, sharey=True, num=name, figsize=(3, 4), tight_layout=True)
        axes = [axes]
        # fig, axes = plt.subplots(ncols=len(tag_cross_seeds) + 1, sharey=True, num=name)
        fig.suptitle(name)
        barplotter.init_axes(axes)

        ### Baseline - Count sequences
        keys = ["0", "not 0", "1", "not 1", "2", "not 2", "all"]
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
        shared_all_seeds = reorder(shared_all_seeds, order)
        ### Plotting
        barplotter.bar_sequences(shared_all_seeds, axes, is_baseline=True)


        ### Patch - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
        shared_all_seeds = reorder(shared_all_seeds, order)
        ### Plotting
        barplotter.bar_sequences(shared_all_seeds, axes)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                   ncol=1, fancybox=True, shadow=True
                   )
        plt.tight_layout()
        pickler.save_figure(f"{name}_across_seeds_{detection_spots}", fig)



def reorder(shared:OrderedDict, order:list) -> OrderedDict:
    ordered = OrderedDict()
    for o in order:
        ordered[o] = shared[o]
    return ordered






if __name__ == '__main__':
    main()
    plt.show()
