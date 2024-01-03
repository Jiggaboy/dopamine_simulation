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

from analysis.sequence_correlation import SequenceCorrelator, BS_TAG, PATCH_TAG
from plot.sequences import imshow_correlations, imshow_correlation_difference
from plot.sequences import plot_db_sequences


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
    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=force_patch, force_baseline=force_baseline)

    _request_plot = UNI.yes_no("Do you want to plot the detected sequences?")
    if _request_plot:
        plot_db_sequences(config, config.get_all_tags())

    for tag_across_seeds in config.get_all_tags(specific_tag, seeds="all"):
        fig, axes = plt.subplots(ncols=len(tag_across_seeds) + 1, nrows=3)
        for t, tag in enumerate(tag_across_seeds):
            imshow_correlations(correlator.correlations[tag][BS_TAG], tag=tag, ax=axes[0, t])
            imshow_correlations(correlator.correlations[tag][PATCH_TAG], is_baseline=False, tag=tag, ax=axes[1, t])
            corr_diff = correlator.correlations[tag][PATCH_TAG] - correlator.correlations[tag][BS_TAG]
            imshow_correlation_difference(corr_diff, ax=axes[2, t])

        # Plot averages
        correlation_avgs = {}
        for i, id_ in enumerate((BS_TAG, PATCH_TAG)):
            corr = [correlator.correlations[tag][id_] for tag in tag_across_seeds]
            correlation_avgs[id_] = np.asarray(corr).mean(axis=0)

        for i, id_ in enumerate((BS_TAG, PATCH_TAG)):
            imshow_correlations(correlation_avgs[id_], tag="Average", ax=axes[i, -1], is_baseline=bool(i))
        corr_diff = correlation_avgs[PATCH_TAG] - correlation_avgs[BS_TAG]
        imshow_correlation_difference(corr_diff, ax=axes[2, -1])


        # break
    plt.show()






if __name__ == '__main__':
    main()
