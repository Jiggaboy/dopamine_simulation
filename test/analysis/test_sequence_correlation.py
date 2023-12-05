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
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable
import unittest as UT

from analysis import sequence_correlation as SC

#===============================================================================
# TEST CLASS
#===============================================================================



class TestDBScan(UT.TestCase):

    def setUp(self):
        self.rnd_gen = np.random.default_rng(seed=1)
        self.correlator = SC.SequenceCorrelator(None)
        self.sigma = 10
        self.dt = .01


    def tearDown(self):
        plt.show()


    # @UT.skip
    def test_correlate_sequences(self):
        shift = 15
        spikes_pre, spikes_post = self._create_pre_post_trains(shift, 2, pre_spikes=0, post_spikes=2)
        self._plot_correlated_spike_trains(spikes_pre, spikes_post)
        self._plot_correlated_spike_trains(spikes_post, spikes_pre)


    def _plot_correlated_spike_trains(self, pre, post):
        fig, (ax1, ax2) = plt.subplots(2)
        for s in (pre, post):
            train, t_axis, kernel = self.correlator._convolve_gauss_kernel(s, self.sigma, self.dt)
            ax1.plot(t_axis, train)
        correlation_normalized, time_axis_correlation = self.correlator._correlate_sequence_times(pre, post)
        ax2.plot(time_axis_correlation, correlation_normalized)

        print(f"Fraction is: {np.max(correlation_normalized) / pre.size}")



    def _create_spike_train(self, spike_count, max_time:float=1000)->np.ndarray:
        return np.sort(self.rnd_gen.integers(0, max_time, size=spike_count))


    def _create_pre_post_trains(self, shift:float, common_spikes:int, pre_spikes:int=0, post_spikes:int=0):
        spikes_pre = self._create_spike_train(common_spikes)
        spikes_post = spikes_pre + shift

        spikes_pre = np.sort(np.append(spikes_pre, self._create_spike_train(pre_spikes)))
        spikes_post = np.sort(np.append(spikes_post, self._create_spike_train(post_spikes)))

        return spikes_pre, spikes_post



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    UT.main()
