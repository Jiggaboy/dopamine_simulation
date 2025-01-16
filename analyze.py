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

from params import config

import lib.universal as UNI
from analysis.sequence_correlation import SequenceCorrelator
import analysis.dbscan_sequences as dbs
from analysis.activity import _average_rate

# Cluster activity
force_patch = UNI.yes_no("Force clustering for patch simulations?", False)
force_baseline = UNI.yes_no("Force baseline clustering?", False)

specific_tag = "repeat"
specific_tag = None


import lib.pickler as PIC
def average_rate(tags, **save_params):
    """Averages the rates of the given tags. Saves the averaged rates."""
    for tag in tags:
        # TODO: TEst for average file -> Not the rate file.
        # if PIC.datafile_exists(tag, **save_params):
        #     continue

        rate = PIC.load_rate(tag, exc_only=True, **save_params)
        avgRate = rate.mean(axis=1)
        PIC.save_avg_rate(avgRate, tag, **save_params)

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    logger.info(f"Average baseline rates: {config.baseline_tags}")
    average_rate(config.baseline_tags, sub_directory=config.sub_dir, config=config)

    tags = config.get_all_tags()
    logger.info(f"Average rates: {tags}")
    average_rate(tags, sub_directory=config.sub_dir, config=config)


    all_tags = config.get_all_tags(specific_tag)
    correlator = SequenceCorrelator(config)


    logger.info("Scan Baselines")
    scanner = dbs.DBScan_Sequences(config)
    for tag in config.baseline_tags:
        spikes, _ = scanner._scan_spike_train(tag, force=force_baseline)

    logger.info("Scan Patches")
    for tag in all_tags:
        spikes, _ = scanner._scan_spike_train(tag, force=force_patch)


    logger.info("Count shared sequences")
    for tag in all_tags:
        try:
            correlator.count_shared_sequences(tag)
        except KeyError:
            logger.info(f"{tag}: No detections spots defined.")
            continue




#===============================================================================
# METHODS
#===============================================================================









if __name__ == '__main__':
    main()
