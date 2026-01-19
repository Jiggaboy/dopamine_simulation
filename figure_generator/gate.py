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
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================

from cflogger import logger

from collections import OrderedDict
import matplotlib.pyplot as plt

from analysis.sequence_correlation import SequenceCorrelator
from params.config_handler import config
from params.motifconfig import GateConfig, SelectConfig
allowed_configs = (GateConfig, SelectConfig)
if type(config) not in allowed_configs:
    logger.info(f"Config: {type(config)}")
    logger.info("No valid config given. Fall back to default.")
    config = GateConfig()

import lib.universal as UNI
import lib.pickler as PIC

from figure_generator.lib import BarPlotter, bar


#===============================================================================
# CONSTANTS
#===============================================================================

gater_tags = "gate-right", "gate-left"
gater_tags = "gate-right-2", "gate-left-2"
# gater_tags = "gate-left",
# gater_tags = "gate-right",



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    all_tags = config.get_all_tags(gater_tags)
    correlator = SequenceCorrelator(config)

    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=False, force_baseline=False)

    for tag in gater_tags:
        plot_gater(config, tag)


#===============================================================================
# GATE
#===============================================================================

def plot_gater(config, tag):
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag)
    tags = config.get_all_tags(tag, seeds="all")

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


    order = [
        "left",
        "left & post",
        "right",
        "right & post",
        "post",
        "left & right",
        "all"
    ]

    labels = [
        r"$B_1$",
        r"$M\,$&$\,B_2$",
        r"$B_2$",
        r"$M\,$&$\,B_1$",
        r"$M$",
        r"$B_1\,$&$\,B_2$",
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
        shared_all_seeds_bs = barplotter.get_sequences_across_seeds(keys, is_baseline=True)
        shared_all_seeds_bs = reorder(shared_all_seeds_bs, order)

        print(tag_cross_seeds)
        for key, value in shared_all_seeds_bs.items():
            print(f"{key}:", value, value.mean(), value.std(ddof=1))
        print()
        ### Plotting
        fig = bar(order, shared_all_seeds_bs, name)

        ### Patch - Count sequences
        shared_all_seeds = barplotter.get_sequences_across_seeds(keys)
        shared_all_seeds = reorder(shared_all_seeds, order)

        for key, value in shared_all_seeds.items():
            print(f"{key}:", value, value.mean(), value.std(ddof=1))
        ### Plotting
        fig = bar(order, shared_all_seeds, name)
        plt.legend(loc="upper right")
        PIC.save_figure(f"{name}_across_seeds_{detection_spots}", fig,
                        sub_directory=config.sub_dir, transparent=True)

        # Variability across seeds in the baseline
        # import numpy as np
        # plt.figure(f"Seq. count:{name}")
        # plt.title(f"Landscape Seed: {config.landscape.seed}")
        # for i, (key, value) in enumerate(shared_all_seeds_bs.items()):
        #     x = np.linspace(i, i+0.8, len(value), endpoint=False)
        #     plt.bar(x - 0.4, value, width=x[1]-i, edgecolor="k", align="edge")
        #     plt.xticks(
        #         np.arange(len(order)), order,
        #         rotation=0,
        #     )
        # break



def reorder(shared:OrderedDict, order:list) -> OrderedDict:
    ordered = OrderedDict()
    for o in order:
        ordered[o] = shared[o]
    return ordered






if __name__ == '__main__':
    main()
    plt.show()
