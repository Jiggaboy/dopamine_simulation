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
import cflogger
logger = cflogger.getLogger()

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable


import lib.pickler as PIC
import universal as UNI
import dopamine as DOP

from analysis import AnalysisFrame, SequenceDetector
from analysis.lib import DBScan
from lib import SequenceCounter, functimer

from params import PerlinConfig, StarterConfig, ScaleupConfig, LowDriveConfig



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def analyze(config:object=None):
    cf = PerlinConfig() if config is None else config
    controls = cf.analysis.dbscan_controls
    scanner = DBScan_Sequences(cf)
    if controls.sequences_across_baselines:
        scanner.sequences_across_baselines(controls.detection_spots)
    if controls.run_dbscan:
        scanner.run_dbscan(controls.detection_spots)
    if controls.sequence_by_cluster:
        scanner.sequence_by_cluster(controls.detection_spots)



#===============================================================================
# CLASS
#===============================================================================

class DBScan_Sequences(AnalysisFrame):

    def __post_init__(self):
        super().__post_init__()
        self._params = self._config.analysis.sequence


    @functimer(logger=logger)
    def sequences_across_baselines(self, tag_spots:list):
        """
        Detects sequences (by individual neurons) in the baseline simulations for all center in 'tag_spots'.
        """
        all_center = UNI.get_center_from_list(tag_spots)
        for bs_tag in self._config.baseline_tags:
            clustered_spikes, _ = self._scan_spike_train(bs_tag)
            self._detect_sequences_dbscan(bs_tag, all_center, clustered_spikes, save=True)


    @functimer(logger=logger)
    def run_dbscan(self, tag_spots:list):
        """
        Detects sequences (by individual neurons) in the tags and the corresponding baseline simulation.

        Saves both to the same object.
        """
        for tag, center in tag_spots:
            for seed in self._config.drive.seeds:
                spikes_bs, _ = self._scan_spike_train(self._config.baseline_tag(seed))
                full_tags = self._config.get_all_tags(tag, seeds=seed)
                for t in full_tags:
                    try:
                        spikes, _ = self._scan_spike_train(tag=t)
                        self._detect_sequences_dbscan(t, center, spikes_bs, spikes, save=True)
                    except FileNotFoundError:
                        logger.error(f"Could not run DBSCAN: File not found for tag: {t}")
                        continue


    def _scan_spike_train(self, tag:str, eps:float=None, min_samples:int=None, plot:bool=False)->(np.ndarray, list):
        """Performs a DBScan on the 'spike train' of the neuronal activity."""
        eps = eps if eps is not None else self._params.eps
        min_samples = min_samples if min_samples is not None else self._params.min_samples

        db = DBScan(eps=eps, min_samples=min_samples)
        spike_train = load_spike_train(self._config, tag)
        # TEST ONLY: Reduce the size of the spike_train
        # spike_train = spike_train[np.logical_and(spike_train[:, 0] > 0, spike_train[:, 0] < 2000)]
        data, labels = db.fit_toroidal(spike_train, nrows=self._config.rows)

        # TODO: plot??? TEST only?
        # if plot:
        #     SUBSAMPLE = 1
        #     _plot_cluster(data[::SUBSAMPLE], labels[::SUBSAMPLE], force_label=None, center = (28, 24))
        return data, labels


    def _detect_sequences_dbscan(self, tag:str, center:list, spikes_bs:np.ndarray, spikes:np.ndarray=None, radius:float=None, save:bool=True)->None:
        """
        Scans the spikes and detect sequences by individual neuronal activity.

        Parameters
        ----------
        tag : str
            Tag to analyze. Used to identify the data set.
        center : list
            Center to analyze.
        spikes_bs : np.ndarray
            The detected spikes in the baseline simulation.
        spikes : np.ndarray, optional
            The detected spikes in the simulation with a NM patch. The default is None.
        radius : float, optional
            Radius of the patch. The default is provided in config object.
        save : bool, optional
            The default is True. If False, it is a dry run.

        Returns
        -------
        None.

        """
        radius = self._params.radius if radius is None else radius
        neurons = self.neurons_from_center(center, radius)

        counter = SequenceCounter(tag, center)

        # times correspond to the spike times; cluster count to the absolut number of clusters
        counter.baseline, counter.baseline_avg = self._cluster_counts(spikes_bs, neurons)
        if spikes is not None:
            counter.patch, counter.patch_avg = self._cluster_counts(spikes, neurons)
        if save:
            PIC.save_db_sequence(counter, counter.tag, sub_directory=self._config.sub_dir)


    def _cluster_counts(self, spikes:np.ndarray, neurons:np.ndarray)->(np.ndarray, int):
        # cluster_count is a 2D array consisting of the detected sequences/clusters for each neuron in each center (determined in neurons)
        times, cluster_count = np.array([self.scan_sequences(spikes, neuron) for neuron in neurons], dtype=object).T
        logger.debug(f"# clusters: {cluster_count}, Spike_times per neuron: {times}")
        return cluster_count, [s.mean() for s in cluster_count]


    @staticmethod
    def get_bins(sim_time:int, bin_width:int):
        """Create the bins for e.g. a histogram."""
        return np.arange(0, sim_time + bin_width, bin_width)


    @staticmethod
    def _detect_sequences_by_cluster(spike_times:np.ndarray, sim_time:int, bin_width:int, peak_threshold:float, min_peak_distance:float)->tuple:
        """
        Determines the bins according to config and binwidth to histogram the times.
        Detects the peaks/no. of sequences using the histogram and return the indeces of the sequences as well as the no. of sequences.

        Return:
            times_indeces, no. of sequences
        """
        bins = DBScan_Sequences.get_bins(sim_time, bin_width)
        hist, _ = np.histogram(np.asarray(spike_times), bins=bins)
        sd = SequenceDetector(None, peak_threshold * bin_width, min_peak_distance)
        # plt.figure("cluster")
        # plt.plot(bins[1:], hist / bin_width)
        return sd._number_of_peaks(hist)


    def neurons_from_center(self, center:list, radius:float)->list:
        patches = [DOP.circular_patch(self._config.rows, c, radius) for c in center]
        neurons = [UNI.patch2idx(patch) for patch in patches]
        return neurons


    def get_cluster_times(self, center:list, spikes:np.ndarray):
        """
        Gets the times which are formed in a cluster in a list for each center (sg.) in the list of center.
        """
        neurons = self.neurons_from_center(center, radius=self._params.radius)
        times, _ = np.array([self.scan_sequences(spikes, neuron) for neuron in neurons], dtype=object).T
        return times


    def sequence_by_cluster(self, tag_spots:list):
        for tag, center in tag_spots:
            for seed in self._config.drive.seeds:
                spikes_bs, _ = self._scan_spike_train(self._config.baseline_tag(seed))
                spike_times_bs = self.get_cluster_times(center, spikes_bs)
                full_tags = self._config.get_all_tags(tag, seeds=seed)
                for t in full_tags:
                    pooled_sequence_times_bs = []
                    pooled_sequence_times = []
                    pooled_no_of_seq_bs = []
                    pooled_no_of_seq = []
                    try:
                        spikes, _ = self._scan_spike_train(t)
                    except FileNotFoundError:
                        logger.info(f"Could not find file for tag: {t}")
                        continue
                    spike_times = self.get_cluster_times(center, spikes)
                    for c, spike_train_bs, spike_train in zip(center, spike_times_bs, spike_times):
                        sequence_times_bs, no_of_seq_bs = self._detect_sequences_by_cluster(spike_train_bs, self._config.sim_time, bin_width=self._params.time_span, peak_threshold=self._params.sequence_threshold, min_peak_distance=self._params.minimal_peak_distance)
                        sequence_times, no_of_seq = self._detect_sequences_by_cluster(spike_train, self._config.sim_time, bin_width=self._params.time_span, peak_threshold=self._params.sequence_threshold, min_peak_distance=self._params.minimal_peak_distance)
                        logger.info(f"Detected sequences: {no_of_seq_bs} to {no_of_seq}")
                        pooled_sequence_times_bs.append(sequence_times_bs)
                        pooled_sequence_times.append(sequence_times)
                        pooled_no_of_seq_bs.append(no_of_seq_bs)
                        pooled_no_of_seq.append(no_of_seq)
                    counter = SequenceCounter(t, center)
                    counter.baseline_times, counter.baseline_avg = pooled_sequence_times_bs, pooled_no_of_seq_bs
                    counter.patch_times, counter.patch_avg = pooled_sequence_times, pooled_no_of_seq
                    PIC.save_db_cluster_sequence(counter, counter.tag, sub_directory=self._config.sub_dir)


    def scan_sequences(self, clustered_rate:np.ndarray, neuron:(int, list))->(list, np.ndarray):
        """
        Scans the pre-scanned spike times provided in {clustered_rate}.
        Determines the spike times and the sequence count independent of individual neurons (loses the information about the neuron which spiked).

        Parameters
        ----------
        clustered_rate : np.ndarray
            Shape (spike_times, spike_details) with spike_details being of shape (t, x, y).
        neuron : (int, list)
            The neurons to take into consideration.

        Returns
        -------
        times : list
            The spike times a sequence is detected.
        seq_counts : np.ndarray
            Detected sequences per neuron.

        """

        neuron = UNI.make_iterable(neuron)
        coordinates = UNI.get_coordinates(self._config.rows)

        seq_counts = np.zeros(len(neuron))
        times = []
        for idx, n in enumerate(neuron):
            coordinate = coordinates[n]

            # find all cluster points which correspond to the coordinate && extract the time points
            # clustered_rate is saved in (spike, details) with details being (t, x, y)
            sequence_activation = (clustered_rate[:, 1:] == coordinate).all(axis=1).nonzero()[0]
            times_sequence = clustered_rate[sequence_activation, 0]
            # TODO: Every distance of >1 means different clusters!!!!
            # np.diff determines the time difference between clusters. Using a single timestep collapses all detected threshold crossing for the same time.
            cluster_count = np.count_nonzero(np.diff(times_sequence) > 1)
            seq_counts[idx] = cluster_count
            times.extend(times_sequence)
        return times, seq_counts



# TODO: either put in class or somewhere else?
def load_spike_train(config:object, tag:str, threshold:float=None):
    """Loads the rate (from tag) and prepares it as a spike train linked to the coordinates of neurons."""
    threshold = config.analysis.sequence.spike_threshold if threshold is None else threshold
    coordinates, rate = PIC.load_coordinates_and_rate(config, tag)
    bin_rate = UNI.binarize_rate(rate.T, threshold)
    return extract_spikes(bin_rate, coordinates)


# TODO: either put in class or somewhere else?
def empty_spike_train(bin_rate:np.ndarray)->np.ndarray:
    """Creates an empty spike  rain with space for coordinates and time points."""
    total_spikes = np.count_nonzero(bin_rate)
    return np.zeros((total_spikes, 3), dtype=int)


# TODO: either put in class or somewhere else?
def extract_spikes(bin_rate:np.ndarray, coordinates:np.ndarray, TD:float=1):
    """
    Takes the time and the coordinations into account and stack them.

    Hyperparameter: TD
    """
    spike_train = empty_spike_train(bin_rate)
    start = end = 0
    for t in range(bin_rate.shape[0]):
        # all spikes S at time t (as indexes)
        S_t = bin_rate[t, :].nonzero()[0]
        spike_count = S_t.size

        end += spike_count
        spike_train[start:end] = np.vstack([np.full(fill_value=t / TD, shape=spike_count), coordinates[S_t].T]).T
        start += spike_count
    return spike_train



# def _plot_cluster(data:np.ndarray, labels:np.ndarray=None, force_label:int=None, center:tuple=None):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 8))
#     ax = plt.axes(projection="3d")
#     ax.set_xlabel("time")
#     ax.set_ylabel("X-Position")
#     ax.set_zlabel("Y-Position")
#     ax.set_ylim(0, 70)
#     ax.set_zlim(0, 70)

#     if center is not None:
#         for x_shift in (-2, 2):
#             ax.plot([data[0, 0], data[-1, 0]], [center[0]+x_shift, center[0]+x_shift], [center[1], center[1]])
#         for y_shift in (-2, 2):
#             ax.plot([data[0, 0], data[-1, 0]], [center[0], center[0]], [center[1]+y_shift, center[1]+y_shift])

#     if labels is None:
#         ax.scatter(*data.T, marker=".")
#         return

#     unique_labels = np.unique(labels)
#     for l in unique_labels:
#         if force_label is not None and l != force_label:
#             continue
#         ax.scatter(*data[labels == l].T, label=l, marker=".")
#     plt.legend()

if __name__ == '__main__':
    analyze()
