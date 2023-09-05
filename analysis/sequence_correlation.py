#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Determines the correlation between two detected sequence series.
    The idea is to relate the effect of a patch to the correlation of sequences.

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

import cflogger
logger = cflogger.getLogger()


from dataclasses import dataclass
import itertools
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from scipy.stats import norm
import scipy.signal as scs
# from collections import namedtuple
# from collections.abc import Iterable

from lib import pickler as PIC
from params import BrianConfig, SelectConfig, GateConfig


SIGMA = 2.
DT = .1

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""
    config = BrianConfig()
    all_tags = config.get_all_tags()
    correlator = SequenceCorrelator(config, sigma=SIGMA, dt=DT)
    for tag in all_tags:
        correlator.correlate_sequences(tag)

    plot_transmission_fraction = input("Plot Transmission fraction? (y/n)").lower() == "y"
    plot_detailed_correlation = input("Plot detailed correlations? (y/n)").lower() == "y"

    if plot_transmission_fraction:
        from plot import sequences




#===============================================================================
# METHODS
#===============================================================================

@dataclass
class SequenceCorrelator:
    config: object
    sigma: float = SIGMA
    dt: float = DT
    t_padding: float = 100.


    def correlate_sequences(self, tag:str):
        sequence = PIC.load_db_cluster_sequence(tag, sub_directory=self.config.sub_dir)
        len_center = len(sequence.center)
        correlations_bs = np.empty(shape=(len_center, len_center, 2), dtype=object)
        correlations_patch = np.empty(shape=(len_center, len_center, 2), dtype=object)
        # Iterate across all combinations
        for pre, post in itertools.permutations(range(len_center), 2):
            # Correlate baseline pre with post
            correlation_bs, time_bs = self._correlate_sequence_times(sequence.baseline_times[pre], sequence.baseline_times[post])
            # Cannot convert to array beforehand as the exact amount of correlated points in not predetermined
            correlations_bs[pre, post] = time_bs, correlation_bs
            correlation, time = self._correlate_sequence_times(sequence.patch_times[pre], sequence.patch_times[post])
            correlations_patch[pre, post] = time, correlation
        sequence.correlations_baseline = correlations_bs
        sequence.correlations_patch = correlations_patch
        logger.info(f"Update sequence object ({tag}) with correlations.")
        PIC.save_db_cluster_sequence(sequence, tag, sub_directory=self.config.sub_dir)


    def _correlate_sequence_times(self, seq_pre:np.ndarray, seq_post:np.ndarray, normalized:bool=True)->tuple:
        """
        Generates a time series from the seq. data using a gaussian kernel. Then cross-correlates the two time series.

        To plot the convolved time series, the method _convolve_gauss_kernel can be used.


        Parameters
        ----------
        seq_pre : np.ndarray
            Sequence TIMES of the pre-patch.
        seq_post : np.ndarray
            Sequence TIMES of the post-patch.
        normalized : bool, optional
            If True, correlation height is normalized to the kernel. Maximum corresponds to the # of correlated sequences.
            The default is True.

        Returns
        -------
        (np.ndarray, np.ndarray):
            Returns the (normalized) correlation of the sequence, as well as the corresponding time series.
            A neg. time lag means, the post sequence is following.


        """
        time_series_pre, time_axis_pre, kernel = self._convolve_gauss_kernel(seq_pre)
        time_series_post, time_axis_post, _ = self._convolve_gauss_kernel(seq_post)
        correlation = scs.correlate(time_series_pre, time_series_post, mode="full")
        normalized_correlation = self._normalize_correlation(correlation, kernel)
        re_corr = normalized_correlation if normalized else correlation
        time_axis_correlation = _correlation_lag(time_axis_pre, time_axis_post, dt=self.dt)
        return re_corr, time_axis_correlation



    def _convolve_gauss_kernel(self, spike_times:np.ndarray, sigma:float=1, dt:float=.1, t_padding:float=100.):
        time_series, time_axis = _spikes_to_timeseries(spike_times, dt=self.dt, padding=self.t_padding)
        kernel = _gauss_kernel(self.sigma, t_width=4*self.sigma, dt=self.dt)
        return np.convolve(kernel, time_series, mode="same"), time_axis, kernel


    @staticmethod
    def _normalize_correlation(correlation:np.ndarray, kernel:np.ndarray):
        return correlation / np.sum(kernel**2)


def _gauss_kernel(sigma:float, t_width:float, dt:float=1.):
    time = np.arange(-t_width, t_width, dt)
    return norm.pdf(time, 0, sigma)


def _correlation_lag(in1:np.ndarray, in2:np.ndarray, dt:float):
    return scs.correlation_lags(len(in1), len(in2), mode="full") * dt


def _spikes_to_timeseries(spikes:np.ndarray, dt:float, padding:float):

    try:
        last_spike = spikes[-1]
    except IndexError:
        last_spike = 0

    bins = np.arange(-padding, last_spike + padding, dt)
    hist, _ = np.histogram(spikes, bins=bins)
    return hist, bins[:-1]




if __name__ == '__main__':
    main()
