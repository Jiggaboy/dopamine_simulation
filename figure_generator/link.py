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

import numpy as np
import matplotlib.pyplot as plt

import lib.pickler as PIC
from params import config
from params.motifconfig import RepeatConfig, GateConfig, LinkConfig
# config = LinkConfig()
from lib import dopamine as DOP
import scipy.signal as ss


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""
    tags_by_seed = config.get_all_tags(None, seeds="all")
    # tags_by_seed = [config.baseline_tags]


    for center in tags_by_seed:
        fig_corr, axes_corr = plt.subplots(nrows=len(center), ncols=3, num=center[0])
        for tag, axis_corr in zip(center, axes_corr):
            print(tag)
            rate = PIC.load_rate(tag, skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)
            bs_rate = PIC.load_rate(config.get_baseline_tag_from_tag(tag), skip_warmup=True, exc_only=True, sub_directory=config.sub_dir, config=config)

            # base 22
            left = DOP.circular_patch(config.rows, center=(38, 48), radius=3, coordinates=config.coordinates)
            # left = DOP.circular_patch(config.rows, center=(9, 53), radius=3, coordinates=config.coordinates)
            # left = DOP.circular_patch(config.rows, center=(58, 32), radius=3, coordinates=config.coordinates)
            right = DOP.circular_patch(config.rows, center=(54, 45), radius=3, coordinates=config.coordinates)
            # right = DOP.circular_patch(config.rows, center=(60, 6), radius=3, coordinates=config.coordinates)

            # base 24
            # left = DOP.circular_patch(config.rows, center=(28, 32), radius=3, coordinates=config.coordinates)
            # right = DOP.circular_patch(config.rows, center=(42, 58), radius=3, coordinates=config.coordinates)
            # right = DOP.circular_patch(config.rows, center=(52, 24), radius=3, coordinates=config.coordinates)

            # # # base repeat
            # left = DOP.circular_patch(config.rows, center=(45, 71), radius=3, coordinates=config.coordinates)
            # right = DOP.circular_patch(config.rows, center=(37, 56), radius=3, coordinates=config.coordinates)

            # # # base 9, GateConfig
            if config.landscape.params["base"] == 9:
                left = DOP.circular_patch(config.rows, center=(41, 61), radius=3, coordinates=config.coordinates)
                # rather right
                left = DOP.circular_patch(config.rows, center=(31, 67), radius=3, coordinates=config.coordinates)
                # right = DOP.circular_patch(config.rows, center=(31, 67), radius=3, coordinates=config.coordinates)
                # rather post
                right = DOP.circular_patch(config.rows, center=(26, 51), radius=3, coordinates=config.coordinates)

            # base 35, SynchroConfig
            if config.landscape.params["base"] == 35:
                left = DOP.circular_patch(config.rows, center=(49, 70), radius=3, coordinates=config.coordinates)
                # left = DOP.circular_patch(config.rows, center=(51, 49), radius=3, coordinates=config.coordinates)
                right = DOP.circular_patch(config.rows, center=(65, 64), radius=3, coordinates=config.coordinates)



            correlate, lags, rate_1_avg, rate_2_avg = correlate_rates(rate, left, right)
            correlate_bs, lags_bs, rate_1_avg_bs, rate_2_avg_bs = correlate_rates(bs_rate, left, right)

            # plt.figure(tag)
            ax = axis_corr[2]
            center = lags.size // 2
            idx = slice(center - 200, center + 200)
            # plt.plot(correlate / right_avg.size / right_avg.mean() / left_avg.mean())
            ax.plot(lags[idx], correlate[idx] / rate_1_avg.size, label="NM", c="r")
            ax.plot(lags_bs[idx], correlate_bs[idx] / rate_1_avg.size, label="bs", c="k")
            # ax.set_xlim(-200, 200)
            ax.legend()

            # plt.figure(tag + "_rate")
            ax = axis_corr[0]
            ax.plot(rate_1_avg, label="left")
            ax.plot(rate_2_avg, label="right")
            ax.legend()
            ax = axis_corr[1]
            ax.plot(rate_1_avg_bs, label="bs_left", c="g")
            ax.plot(rate_2_avg_bs, label="bs_right", c="c")
            ax.legend()


    plt.show()

#===============================================================================
# METHODS
#===============================================================================

def correlate_rates(rate, left, right):
    rate_1 = rate[left]
    rate_2 = rate[right]
    rate_1_avg = rate_1.mean(axis=0)
    rate_2_avg = rate_2.mean(axis=0)
    scale1 = np.max(rate_1_avg)
    scale2 = np.max(rate_2_avg)
    scale = max(scale1, scale2)
    rate_1_avg -= rate_1_avg.mean()
    rate_2_avg -= rate_2_avg.mean()
    # rate_1_avg /= rate_1_avg.std()
    # rate_2_avg /= rate_2_avg.std()
    rate_1_avg /= scale
    rate_2_avg /= scale
    correlate = ss.correlate(rate_1_avg, rate_2_avg, mode="same")# / rate_1_avg.std() / rate_2_avg.std()
    lags = ss.correlation_lags(rate_1_avg.size, rate_2_avg.size, mode="same")

    return correlate, lags, rate_1_avg, rate_2_avg










if __name__ == '__main__':
    main()
