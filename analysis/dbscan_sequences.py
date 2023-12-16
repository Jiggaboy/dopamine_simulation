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

import lib.pickler as PIC
import lib.universal as UNI

from analysis.lib import AnalysisFrame, DBScan
from lib import SequenceCounter, functimer
from plot.sequences import plot_db_sequences, plot_baseline_sequences

from params import config


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    analyze(config)


def analyze(config:object=None):
    controls = config.analysis.dbscan_controls
    scanner = DBScan_Sequences(config)

    force_cluster_sequences_across_patches = UNI.yes_no("Force cluster analysis patches? (y/n)")
    scanner.sequence_by_cluster(controls.detection_spots, force=force_cluster_sequences_across_patches)


    _request_plot = input("Do you want to plot the detected sequences? (y: all; p:patches only; bs:baselines only)").lower()
    if _request_plot == "y":
        plot_baseline_sequences(config)
        plot_db_sequences(config, config.get_all_tags())
    elif _request_plot == "p":
        plot_db_sequences(config, config.get_all_tags())
    elif _request_plot == "bs":
        plot_baseline_sequences(config)
    import matplotlib.pyplot as plt
    plt.show()



#===============================================================================
# CLASS
#===============================================================================

class DBScan_Sequences(AnalysisFrame):

    def __post_init__(self):
        super().__post_init__()
        self._params = self._config.analysis.sequence # Load analysis parameter


    def _scan_spike_train(self, tag:str, eps:float=None, min_samples:int=None, plot:bool=False, force:bool=False)->(np.ndarray, list):
        """Performs a DBScan on the 'spike train' of the neuronal activity."""
        eps = eps if eps is not None else self._params.eps
        min_samples = min_samples if min_samples is not None else self._params.min_samples

        identifier, filename = self._get_spike_train_identifier_filename(tag, eps, min_samples)
        if not force:
            try:
                obj = PIC.load_spike_train(filename, sub_directory=self._config.sub_dir)
                logger.info(f"Load spike train of tag: {tag}")
                return obj["data"], obj["labels"]
            except FileNotFoundError:
                pass
        return self._sweep_spike_train(tag, eps, min_samples, save=True)


    @functimer(logger=logger)
    def _sweep_spike_train(self, tag:str, eps:float=None, min_samples:int=None, save:bool=True)->(np.ndarray, list):
        db = DBScan(eps=eps, min_samples=min_samples, n_jobs=-1)
        spike_train = load_spike_train(self._config, tag)
        data, labels = db.fit_toroidal(spike_train, nrows=self._config.rows)

        if save:
            identifier, filename = self._get_spike_train_identifier_filename(tag, eps, min_samples)
            identifier["data"] = data
            identifier["labels"] = labels
            PIC.save_spike_train(identifier, filename, sub_directory=self._config.sub_dir)
        return data, labels


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
        # sd = SequenceDetector(None, peak_threshold * bin_width, min_peak_distance)
        return UNI.get_peaks(hist, peak_threshold * bin_width, min_peak_distance, bin_width)


    @functimer(logger=logger)
    def sequence_by_cluster(self, tag_spots:list, **spike_kwargs):
        # The tags are defined by the analysis parameters, not the tags in the config.
        for tag, center in tag_spots:
            for seed in self._config.drive.seeds:
                spikes_bs = None
                full_tags = self._config.get_all_tags(tag, seeds=seed)
                for t in full_tags:
                    logger.info(f"Analyzing {t}...")
                    # Load patch spike times.
                    try:
                        spikes, spike_labels = self._scan_spike_train(t, **spike_kwargs)
                    except FileNotFoundError:
                        logger.info(f"Could not find file for tag: {t}")
                        continue
                    # Load baseline if not done yet
                    if spikes_bs is None:
                        spikes_bs, spike_labels_bs = self._scan_spike_train(self._config.baseline_tag(seed), **spike_kwargs)

                    counter = SequenceCounter(t, center)
                    # Add the actual spikes here
                    counter.baseline_spikes = spikes_bs
                    counter.patch_spikes = spikes

                    spike_labels_bs = self.squeeze_labels(spike_labels_bs)
                    spike_labels = self.squeeze_labels(spike_labels)

                    counter.baseline_labels = spike_labels_bs
                    counter.patch_labels = spike_labels
                    PIC.save_db_cluster_sequence(counter, counter.tag, sub_directory=self._config.sub_dir)


    @staticmethod
    @functimer(logger=logger)
    def squeeze_labels(labels:np.ndarray):
        unique_labels = set(labels)
        for i, label in enumerate(sorted(unique_labels)):
            labels[labels == label] = i
        return labels


    @staticmethod
    def _get_spike_train_identifier_filename(tag, eps, min_samples):
        identifier = {
            "tag": tag,
            "eps": str(eps),
            "min_samples": str(min_samples),
        }
        filename = "_".join(identifier.values())
        return identifier, filename



    @staticmethod
    def get_bins(sim_time:int, bin_width:int):
        """Create the bins for e.g. a histogram."""
        return np.arange(0, sim_time + bin_width, bin_width)


# TODO: either put in class or somewhere else?
def load_spike_train(config:object, tag:str, threshold:float=None):
    """Loads the rate (from tag) and prepares it as a spike train linked to the coordinates of neurons."""
    threshold = config.analysis.sequence.spike_threshold if threshold is None else threshold
    coordinates, rate = PIC.load_coordinates_and_rate(config, tag)
    bin_rate = UNI.binarize_rate(rate.T, threshold)
    return extract_spikes(bin_rate, coordinates)


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


# TODO: either put in class or somewhere else?
def empty_spike_train(bin_rate:np.ndarray)->np.ndarray:
    """Creates an empty spike  rain with space for coordinates and time points."""
    total_spikes = np.count_nonzero(bin_rate)
    return np.zeros((total_spikes, 3), dtype=int)


if __name__ == '__main__':
    main()
