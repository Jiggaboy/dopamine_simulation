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
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from cflogger import logger

# import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from analysis.sequence_correlation import SequenceCorrelator
from params.config_handler import config
from params.motifconfig import SelectConfig
allowed_configs = (SelectConfig, )
if type(config) not in allowed_configs:
    logger.info(f"Config: {type(config)}")
    logger.info("No valid config given. Fall back to default.")
    config = SelectConfig()
import lib.universal as UNI
import lib.pickler as PIC

from figure_generator.lib import BarPlotter, bar



#===============================================================================
# CONSTANTS
#===============================================================================

select_tags = "select-left", "select-right"



#===============================================================================
# MAIN METHOD
#===============================================================================
def main():

    all_tags = config.get_all_tags(None)
    correlator = SequenceCorrelator(config)

    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=False, force_baseline=False)
    for tag in select_tags:
        plot_selecter(config, tag)


def plot_selecter(config, tag):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag)
    tags = config.get_all_tags(tag, seeds="all")

    # left, right, merged
    labels = [
        "base",
        "left & right",
        "left",
        "base & right",
        "right",
        "base & left",
        "all"
    ]

    order = [
        "base",
        "base & left",
        "left",
        "base & right",
        "right",
        "left & right",
        "all"
    ]
    labels = [
        r"$M$",
        r"$B_1\,$&$\,B_2$",
        r"$B_1$",
        r"$M\,$&$\,B_2$",
        r"$B_2$",
        r"$M\,$&$\,B_1$",
        r"all"
    ]

    order = [
        r"$M$",
        r"$M\,$&$\,B_1$",
        r"$B_1$",
        r"$M\,$&$\,B_2$",
        r"$B_2$",
        r"$B_1\,$&$\,B_2$",
        r"all"
    ]


    for tag_cross_seeds in tags:
        barplotter = BarPlotter(config, tag_cross_seeds, labels, detection_spots)

        name, _ = UNI.split_seed_from_tag(tag_cross_seeds[0])

        ### Baseline - Count sequences
        keys = ["0", "not 0", "1", "not 1", "2", "not 2", "all"]
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
        shared_all_seeds = reorder(shared_all_seeds, order)

        print(tag_cross_seeds)
        for key, value in shared_all_seeds.items():
            print(f"{key}:", value, value.mean(), value.std(ddof=1))
        print()
        ### Plotting
        fig = bar(order, shared_all_seeds, name)


        ### Patch - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
        shared_all_seeds = reorder(shared_all_seeds, order)


        for key, value in shared_all_seeds.items():
            print(f"{key}:", value, value.mean(), value.std(ddof=1))
        print()
        ### Plotting
        fig = bar(order, shared_all_seeds, name)
        plt.legend(loc="upper right",
                   ncol=1, fancybox=True, shadow=True,
                   )

        PIC.save_figure(f"{name}_across_seeds_{detection_spots}", fig,
                        sub_directory=config.sub_dir, transparent=True)



def reorder(shared:OrderedDict, order:list) -> OrderedDict:
    ordered = OrderedDict()
    for o in order:
        ordered[o] = shared[o]
    return ordered






if __name__ == '__main__':
    main()
    plt.show()
