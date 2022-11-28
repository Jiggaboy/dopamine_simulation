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
import unittest as UT
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable



from analysis.lib.dbscan import DBScan
from analysis import DBScan_Sequences


PLOT_RAMPING_DATA = False


#===============================================================================
# TESTCLASS
#===============================================================================


class TestDBScanSequences(UT.TestCase):
    max_neurons = 12
    test_span = range(2, 5)
    test_threshold = range(3, 6)

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        plt.show()

    @UT.skipIf(not PLOT_RAMPING_DATA, "Plot is not necessary.")
    def test_plot_ramping_data(self):
        data = self._create_data()
        data_re = np.repeat(np.arange(data.size), data.astype(int))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        ax1.hist(data)
        ax2.plot(data)
        hist, bins = np.histogram(data_re, bins=np.arange(data.size))
        ax3.hist(data_re, bins=np.arange(data.size))
        ax4.plot(data_re)


    # @UT.skip
    def test_scan_ramp(self):
        data = self._create_data(add_natural_data=True)
        data = data[200:]
        data_smooth = np.where(data >= self.test_threshold[0], data, 0)
        spikes = self._reverse_hist(data)
        fig, axes = plt.subplots(len(self.test_span))
        for span, ax in zip(self.test_span, axes):
            for t in self.test_threshold:
                idx, seq_count = DBScan_Sequences._detect_sequences_by_cluster(spikes, data.size, span, peak_threshold=t, min_peak_distance=12)
                ax.plot(idx * span, np.full(idx.size, fill_value=t), marker="*", ls="None", label=seq_count)
            ax.plot(data_smooth, "k", zorder=-1)
            ax.legend()


    @UT.skip
    def test_db_params(self):
        from params import PerlinConfig

        force_scan = True
        # force_scan = False
        force_plot = True
        # force_plot = False

        self._config = PerlinConfig()
        center = (28, 24) ## Example center - cf. 'in' center

        epsilons = np.arange(3, 10, 2)
        epsilons = np.arange(2, 5, 2)
        min_samples = np.arange(10, 30, 15, dtype=int)
        # min_samples = np.arange(20, 22, 3, dtype=int)

        scanner = DBScan_Sequences(self._config)
        for eps in epsilons:
            for min_sample in min_samples:
                if force_scan:
                    scanner.test_scan(center, eps=eps, min_samples=min_sample)

                if force_plot:
                    print(f"Scan: {eps}_{min_sample}")
                    spike_train = np.load(f"test_train_{eps}_{min_sample}.npy")
                    fig, axes = plt.subplots(len(self.test_span), num=F"{eps}_{min_sample}")
                    max_idx = spike_train.max() + 100
                    for span, ax in zip(self.test_span, axes):
                        for t in self.test_threshold:
                            idx, seq_count = DBScan_Sequences._detect_sequences_by_cluster(spike_train, max_idx, span, peak_threshold=t, min_peak_distance=12)
                            ax.plot(idx * span, np.full(idx.size, fill_value=t), marker="*", ls="None", label=seq_count)
                        ax.set_xlim(0, 2000)
                        # ax.hist(spike_train, color="k", bins=np.arange(max_idx), zorder=-1)
                        # ax.legend()




    @staticmethod
    def _reverse_hist(data):
        return np.repeat(np.arange(data.size), data.astype(int))

    def _create_data(self, add_ramp:bool=False, add_ramp_with_pauses:bool=False, add_slow_ramp:bool=False, add_mixed_ramp:bool=False, add_mixed_noisy_ramp:bool=False, add_natural_data:bool=False, add_noise:bool=False):
        data = get_pause()
        if add_ramp:
            for n in range(1, self.max_neurons+1):
                data = self.__ramps(data, n)
                data = self.__ramps(data, n, add_downramp=True)

        if add_ramp_with_pauses:
            for p in range(1, 40):
                data = self.__ramps_with_pauses(data, self.max_neurons, pause_length=p)
                data = self.__ramps_with_pauses(data, self.max_neurons, pause_length=p, down_ramp=True)

        if add_slow_ramp:
            for r in range(1, 15):
                for n in range(self.max_neurons):
                    data = self.__slow_ramp(data, n, reps=r)
                    data = self.__slow_ramp(data, n, reps=r, down_ramp=True)
                    data = self.__slow_ramp(data, n, reps=r)
                    data = self.__slow_ramp(data, n, reps=r, down_ramp=True, prepend_pause=False)

        if add_mixed_ramp:
            for r in range(1, 4):
                data = self.__mixed_ramp(data, self.max_neurons, reps=r)
                data = self.__mixed_ramp(data, self.max_neurons, reps=r, down_ramp=True)

                data = self.__mixed_ramp(data, self.max_neurons, reps=r)
                data = self.__mixed_ramp(data, self.max_neurons, reps=r, down_ramp=True, prepend_pause=False)


        if add_mixed_noisy_ramp:
            noise_pattern = [-1, 1]
            for r in range(1, 5):
                data = self.__mixed_ramp(data, self.max_neurons, reps=r, noise_pattern=noise_pattern)
                data = self.__mixed_ramp(data, self.max_neurons, reps=r, down_ramp=True, noise_pattern=noise_pattern)

                data = self.__mixed_ramp(data, self.max_neurons, reps=r, noise_pattern=noise_pattern)
                data = self.__mixed_ramp(data, self.max_neurons, reps=r, down_ramp=True, noise_pattern=noise_pattern, prepend_pause=False)

        if add_natural_data:
            data = self.__natural(data, self.max_neurons)

        if add_noise:
            data = self.__noise(data, self.max_neurons)
        return data


    @staticmethod
    def __ramps(data:np.ndarray, max_height:int, add_downramp:bool=False, prepend_pause:bool=True):
        ramps = np.zeros(1)

        ramp = np.arange(max_height+1)
        ramps = np.append(ramps, ramp)
        if add_downramp:
            down_ramp = np.arange(max_height-1, -1, -1)
            ramps = np.append(ramps, down_ramp)

        if prepend_pause:
            ramps = np.append(get_pause(), ramps)
        return np.append(data, ramps)


    @staticmethod
    def __ramps_with_pauses(data:np.ndarray, max_height:int, pause_length:int=1, down_ramp:bool=False, prepend_pause:bool=True):
        ramp = np.arange(max_height) if not down_ramp else np.arange(max_height-1, -1, -1)
        ramp_with_pauses = np.vstack([ramp, np.zeros((pause_length, max_height))]).T.flatten()

        if prepend_pause:
            ramp_with_pauses = np.append(get_pause(), ramp_with_pauses)
        return np.append(data, ramp_with_pauses)


    @staticmethod
    def __slow_ramp(data:np.ndarray, max_height:int, reps:int=1, down_ramp:bool=False, prepend_pause:bool=True):
        ramp = np.arange(max_height) if not down_ramp else np.arange(max_height-1, -1, -1)
        slow_ramp = np.repeat(ramp, reps)

        if prepend_pause:
            slow_ramp = np.append(get_pause(), slow_ramp)
        return np.append(data, slow_ramp)


    @staticmethod
    def __mixed_ramp(data:np.ndarray, max_height:int, reps:int=1, down_ramp:bool=False, prepend_pause:bool=True, noise_pattern:np.ndarray=None):
        ramp = np.arange(max_height) if not down_ramp else np.arange(max_height-1, -1, -1)
        ramp = np.repeat(ramp, reps)
        mixed_ramp = np.vstack([ramp, ramp+1, ramp]).T.flatten()
        if noise_pattern is not None:
            noise = np.resize(noise_pattern, mixed_ramp.size)
            mixed_ramp += noise
            mixed_ramp = np.maximum(mixed_ramp, 0)
        if prepend_pause:
            mixed_ramp = np.append(get_pause(), mixed_ramp)
        return np.append(data, mixed_ramp)


    @staticmethod
    def __natural(data:np.ndarray, prepend_pause:bool=True):
        import os
        cwd = os.path.dirname(__file__)
        filename = os.path.join(cwd, "natural_spike_train.npy")
        natural = np.load(filename)

        if prepend_pause:
            natural = np.append(get_pause(), natural)
        return np.append(data, natural)


    @staticmethod
    def __noise(data:np.ndarray, max_height:int, seed:int=0, length:int=1000, prepend_pause:bool=True):
        random_generator = np.random.default_rng(seed)
        noise = random_generator.integers(0, max_height+1, size=length)

        if prepend_pause:
            noise = np.append(get_pause(), noise)
        return np.append(data, noise)



def get_pause():
    return np.zeros(100)


if __name__ == '__main__':
    UT.main()
