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

from analysis import AnalysisFrame
from analysis.lib import DBScan
from lib import SequenceCounter

from params import PerlinConfig, StarterConfig, ScaleupConfig


center_in_41 = ((30, 18), (28, 26), )
center_linker_41 = ((21, 65), (67, 30), (30, 66))
center_repeater_41 = ((9, 37), (2, 32), (55, 49))
center_activator_41 = ((67, 30), (50, 37), (60, 46))
center_starter_41 = ((46, 3), (49, 6), (43, 13))


detection_spots_41 = []
UNI.append_spot(detection_spots_41, "in", center_in_41)
UNI.append_spot(detection_spots_41, "linker", center_linker_41)
UNI.append_spot(detection_spots_41, "repeater", center_repeater_41)
UNI.append_spot(detection_spots_41, "out-activator", center_activator_41)
UNI.append_spot(detection_spots_41, "starter", center_starter_41)

detection_spots = detection_spots_41



# SEQ_DETECTION_SPOTS = []
# def prepare_analysis():
#     center = ((30, 18), (28, 26), )
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "in", (center))
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "edge", (center)
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "out", (center)

#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "linker", ((21, 65), (67, 30), (30, 66)))
#     UNI.append_spot(SEQ_DETECTION_SPOTS, "repeater", ((9, 37), (2, 32), (55, 49)))
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "repeater-proxy", ((2, 31), (29, 35), (29, 25)))

#     center = ((67, 30), (50, 37), (60, 46))
#     #center = ((35, 49), (49, 36), (29, 35), (29, 25))
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "in-activator", (center))
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "edge-activator", (center))
#     UNI.append_spot(SEQ_DETECTION_SPOTS, "out-activator", (center))
#     #UNI.append_spot(SEQ_DETECTION_SPOTS, "activator-proxy", (center))

#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "starter", ((46, 3), (49, 6), (43, 13)))
#     # UNI.append_spot(SEQ_DETECTION_SPOTS, "starter", ((48, 8), ))


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    cf = PerlinConfig()
    # prepare_analysis()
    scanner = DBScan_Sequences(cf)
    scanner.sequences_across_baselines(detection_spots)


#===============================================================================
# CLASS
#===============================================================================

class DBScan_Sequences(AnalysisFrame):

    def __post_init__(self):
        super().__post_init__()
        self._params = self._config.analysis.sequence


    def sequences_across_baselines(self, tag_spots:list):
        """
        Deects sequences (by individual neurons) in the baseline simulations for all center in 'tag_spots'.
        """
        all_center = UNI.get_center_from_list(tag_spots)
        for bs_tag in self._config.baseline_tags:
            clustered_spikes, _ = self._scan_spike_train(bs_tag)
            self.detect_sequences_dbscan(bs_tag, all_center, clustered_spikes)


    def _scan_spike_train(self, tag:str)->(np.ndarray, list):
        """Performs a DBScan on the 'spike train' of the neuronal activity."""
        db = DBScan(eps=self._params.eps, min_samples=self._params.min_samples)
        spike_train = load_spike_train(self._config, tag)
        data, labels = db.fit_toroidal(spike_train, nrows=self._config.rows)
        return data, labels


    def detect_sequences_dbscan(self, tag:str, center:list, spikes_bs:np.ndarray, spikes:np.ndarray=None, radius:float=None):
        radius = self._params.radius if radius is None else radius
        patches = [DOP.circular_patch(self._config.rows, c, radius) for c in center]
        neurons = [UNI.patch2idx(patch) for patch in patches]

        counter = SequenceCounter(tag, center)

        # times correspond to the spike times
        # cluster count to the absolut number of clusters
        times_bs, cluster_count_bs = np.array([scan_sequences(self._config, spikes_bs, neuron) for neuron in neurons], dtype=object).T
        counter.baseline, counter.baseline_avg = cluster_count_bs, [s.mean() for s in cluster_count_bs]
        logger.debug(f"Center: {center}, # clusters: {cluster_count_bs}, Spike_times per neuron: {times_bs}")

        if spikes is not None:
            times, cluster_count = np.array([scan_sequences(self._config, spikes, neuron) for neuron in neurons], dtype=object).T
            counter.patch, counter.patch_avg = cluster_count, [s.mean() for s in cluster_count]
            logger.debug(f"Center: {center}, # clusters: {cluster_count}, Spike_times per neuron: {times}")

        PIC.save_db_sequence(counter, counter.tag, sub_directory=self._config.sub_dir)

    #### TODO: Only for checking the method
    def sequence_by_cluster(self, tag_spots:list):
        """
        """
        for tag in self._config.baseline_tags[:3]:
            spikes_bs, _ = dbscan(config, tag=tag)
            for name, center in tag_spots:
                times = get_cluster_times(config, center, spikes_bs)


                figname = f"{tag}_{name}"
                fig, (*ax_times, ax_sequences) = plt.subplots(ncols=4, num=figname, figsize=(12, 6))
                T_SPANS = np.arange(1, 4)
                for i, T_SPAN in enumerate(T_SPANS):
                    logger.info(f"{tag}_{name} with span {T_SPAN}")
                    ax_time = ax_times[i]
                    ax_time.set_title(f"Bin width/Time step: {i + 1}")
                    for c, time in enumerate(times):
                        THRESHOLD = np.arange(1, 10)
                        sequences = np.zeros(shape=(THRESHOLD.shape))
                        for j, T in enumerate(THRESHOLD):
                            idx, no_of_seq = detect_sequence_by_cluster(time, config, bin_width=T_SPAN, peak_threshold=T, min_peak_distance=MINIMAL_PEAK_DISTANCE)
                            sequences[j] = no_of_seq
                            ax_time.plot(idx * T_SPAN, np.full(no_of_seq, fill_value=T * T_SPAN), ms=5, ls="None", marker="o", label=no_of_seq)
                        ax_time.hist(time, bins=get_bins(config, bin_width=T_SPAN))
                        lbl = f"{center[c]} (span:{T_SPAN})"
                        # ax_sequences.plot(THRESHOLD, sequences, label=lbl)
                        _plot_clustered_sequence(THRESHOLD, sequences, axis=ax_sequences, label=lbl)
                        ax_time.set_xlabel("time")
                        ax_time.set_ylabel("# spikes")
                # ax_time.legend()
                ax_sequences.set_xlabel("# neuron threshold")
                ax_sequences.set_ylabel("# Sequences")
                ax_sequences.legend()
                PIC.save_figure(f"seq_by_cluster_{tag}", fig, config.sub_dir)


def scan_sequences(config:object, clustered_rate:np.ndarray, neuron:(int, list)):
    try:
        scan_sequences.pop
    except AttributeError:
        from custom_class import Population
        scan_sequences.pop = Population(config)

    neuron = UNI.make_iterable(neuron)

    seq_counts = np.zeros(len(neuron))
    times = []
    for idx, n in enumerate(neuron):
        coordinate = scan_sequences.pop.coordinates[n]

        # find all cluster points which correspond to the coordinate && extract the time points
        sequence_acitvation = (clustered_rate[:, 1:] == coordinate).all(axis=1).nonzero()[0]
        times_sequence = clustered_rate[sequence_acitvation, 0]
        cluster_count = np.count_nonzero(np.diff(times_sequence) > 1)
        seq_counts[idx] = cluster_count
        times.extend(times_sequence)
    return times, seq_counts



def load_spike_train(config:object, tag:str, threshold:float=None):
    """Loads the rate (from tag) and prepares it as a spike train linked to the coordinates of neurons."""
    threshold = config.analysis.sequence.spike_threshold if threshold is None else threshold
    coordinates, rate = PIC.load_coordinates_and_rate(config, tag)
    bin_rate = UNI.binarize_rate(rate.T, threshold)
    return extract_spikes(bin_rate, coordinates)


def empty_spike_train(bin_rate:np.ndarray)->np.ndarray:
    """Creates an empty spike  rain with space for coordinates and time points."""
    total_spikes = np.count_nonzero(bin_rate)
    return np.zeros((total_spikes, 3), dtype=int)


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


if __name__ == '__main__':
    main()
