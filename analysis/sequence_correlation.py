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

from scipy.stats import norm
import scipy.signal as scs

from lib import pickler as PIC
from lib import universal as UNI
from params import BrianConfig, SelectConfig, GateConfig, GateRepeatConfig, RandomLocationConfig


SIGMA = 4.
DT = .1

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""
    config = RandomLocationConfig()
    all_tags = config.get_all_tags("repeater")
    correlator = SequenceCorrelator(config, sigma=SIGMA, dt=DT)
    for tag in all_tags:
        print(tag)
        correlator.count_shared_sequences(tag)
        # break

    plt.show()
    return
    config = GateRepeatConfig()
    all_tags = config.get_all_tags()
    correlator = SequenceCorrelator(config, sigma=SIGMA, dt=DT)
    for tag in all_tags:
        correlator.correlate_sequences(tag)

    plot_transmission_fraction = input("Plot Transmission fraction? (y/n)").lower() == "y"
    plot_detailed_correlation = input("Plot detailed correlations? (y/n)").lower() == "y"

    if plot_transmission_fraction:
        from plot import sequences
        for tags in config.get_all_tags(seeds="all"):
            sequences.plot_sequence_correlations(config, tags, add_detailed_plot=plot_detailed_correlation)
        plt.show()


from mpl_toolkits.mplot3d import axes3d
def _plot_cluster(data:np.ndarray, labels:np.ndarray=None, force_label:int=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("time")
    ax.set_ylabel("X-Position")
    ax.set_zlabel("Y-Position")

    if labels is None:
        ax.scatter(*data.T, marker=".")
        return

    unique_labels = np.unique(labels)
    print(unique_labels)
    for l in unique_labels:
        if force_label is not None and l != force_label:
            continue
        ax.scatter(*data[labels == l].T, label=l, marker=".")
    plt.legend()



#===============================================================================
# METHODS
#===============================================================================
import lib.dopamine as DOP

@dataclass
class SequenceCorrelator:
    config: object
    sigma: float = SIGMA
    dt: float = DT
    t_padding: float = 1000.



    def neurons_from_center(self, center:list, radius:float)->list:
        patches = [DOP.circular_patch(self.config.rows, c, radius) for c in center]
        neurons = [UNI.patch2idx(patch) for patch in patches]
        return neurons


    def has_spikes_at_center(self, spikes_in_sequence:np.ndarray, center:tuple, coordinates:np.ndarray) -> bool:
        radius = 2
        has_spikes = False

        neurons_at_center = self.neurons_from_center(center, radius)
        neurons = UNI.make_iterable(neurons_at_center[0])
        for idx, n in enumerate(neurons):
            coordinate = coordinates[n]

            spikes_at_center = (spikes_in_sequence[:, 1:] == coordinate).all(axis=1).nonzero()[0]
            spike_times = spikes_in_sequence[spikes_at_center, 0]
            if any(np.diff(spike_times) > 10):
                print(spike_times)
                _plot_cluster(spikes_in_sequence)
                plt.figure()
                sc = plt.scatter(*spikes_in_sequence[:, 1:].T, c=spikes_in_sequence[:, 0], marker="o")
                plt.xlim(0, 80)
                plt.ylim(0, 80)
                plt.colorbar(sc)
                plt.show()
                print(center)
                # raise LookupError
            if spikes_at_center.size > 0:
                has_spikes = True
                break
        return has_spikes

    def get_sequences_id_at_location(self, labels:np.ndarray, spikes:np.ndarray, centers:tuple, coordinates:np.ndarray):
        label_identifier = sorted(set(labels))

        sequence_at_center = np.zeros((len(label_identifier), len(centers)), dtype=bool)

        for label in label_identifier[:]:
            label_idx = labels == label
            spikes_in_sequence = spikes[label_idx]

            # Find those labels, which cross a location
            for c in range(len(centers)):
                center = centers[c:c+1] # retrieve as tuple as neurons from center takes tuples
                sequence_at_center[label, c] = self.has_spikes_at_center(spikes_in_sequence, center, coordinates)
        return sequence_at_center



    def count_shared_sequences(self, tag:str):
        # radius = 2
        coordinates = UNI.get_coordinates(self.config.rows)

        # Load the spike train and its labels
        logger.info("Load spike train and labels")
        sequence = PIC.load_db_cluster_sequence(tag, sub_directory=self.config.sub_dir)
        # Attributes are *_labels (1D), *_spikes (3D), *_times (1D for each location)
        sequence_at_center = self.get_sequences_id_at_location(sequence.baseline_labels, sequence.baseline_spikes,
                                                               sequence.center, coordinates)
        sequence_at_center_patch = self.get_sequences_id_at_location(sequence.patch_labels, sequence.patch_spikes,
                                                               sequence.center, coordinates)

            # if sequence_at_center[label].any():
            #     # Control: Plot the sequence coordinates
            #     plt.figure()
            #     sc = plt.scatter(*spikes_in_sequence[:, 1:].T, c=spikes_in_sequence[:, 0], marker="o")
            #     plt.xlim(0, 80)
            #     plt.ylim(0, 80)
            #     plt.colorbar(sc)
            #     plt.show()

        test = sequence_at_center * np.arange(1, len(sequence.center)+1)
        test_patch = sequence_at_center_patch * np.arange(1, len(sequence.center)+1)
        plt.figure()
        plt.plot(test)
        plt.figure()
        plt.plot(test_patch)
        plt.show()

        # plot ths histogram for each neuron the distance between spikes in time
        # Filter?


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
        correlation = scs.correlate(time_series_pre, time_series_post, mode="same")
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
    return scs.correlation_lags(len(in1), len(in2), mode="same") * dt


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
