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

from analysis.sequence_correlation import SequenceCorrelator


from lib import universal as UNI

specific_tag = "repeat"
specific_tag = None

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    all_tags = config.get_all_tags(specific_tag)
    correlator = SequenceCorrelator(config)

    force_patch = UNI.yes_no("Force clustering for patch simulations?")
    force_baseline = UNI.yes_no("Force baseline clustering?")
    import analysis.dbscan_sequences as dbs
    scanner = dbs.DBScan_Sequences(config)
    for tag in config.baseline_tags:
        spikes, _ = scanner._scan_spike_train(tag)

    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=force_patch, force_baseline=force_baseline)
    plt.show()






if __name__ == '__main__':
    main()
