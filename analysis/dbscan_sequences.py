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

from lib import functimer
import lib.pickler as PIC
import lib.universal as UNI

from analysis.lib import AnalysisFrame, DBScan


#===============================================================================
# CLASS
#===============================================================================

class DBScan_Sequences(AnalysisFrame):

    def __post_init__(self):
        super().__post_init__()
        self._params = self._config.analysis.sequence # Load analysis parameter


    def _scan_spike_train(self, tag:str, eps:float=None, min_samples:int=None, force:bool=False)->(np.ndarray, list):
        """Performs a DBScan on the 'spike train' of the neuronal activity."""
        eps = eps if eps is not None else self._params.eps
        min_samples = min_samples if min_samples is not None else self._params.min_samples

        identifier, filename = PIC.get_spike_train_identifier_filename(tag, eps, min_samples)
        logger.info(f"Scan spike train of: {identifier}...")
        if not force:
            try:
                return PIC.load_spike_train(filename, config=self._config)
            except FileNotFoundError:
                pass
        return self._sweep_spike_train(tag, eps, min_samples, save=True)


    @functimer(logger=logger)
    def _sweep_spike_train(self, tag:str, eps:float=None, min_samples:int=None, save:bool=True)->(np.ndarray, list):
        db = DBScan(eps=eps, min_samples=min_samples, n_jobs=-1)
        spike_train = load_spike_train(self._config, tag)
        data, labels = db.fit_toroidal(spike_train, nrows=self._config.rows)
        labels = self.squeeze_labels(labels)

        if save:
            PIC.save_spike_train(tag, self._config, data, labels)
        return data, labels


    @staticmethod
    def squeeze_labels(labels:np.ndarray):
        unique_labels = set(labels)
        for i, label in enumerate(sorted(unique_labels)):
            labels[labels == label] = i
        return labels



    @staticmethod
    def get_bins(sim_time:int, bin_width:int):
        """Create the bins for e.g. a histogram."""
        return np.arange(0, sim_time + bin_width, bin_width)


# TODO: either put in class or somewhere else?
def load_spike_train(config:object, tag:str, threshold:float=None):
    """Loads the rate (from tag) and prepares it as a spike train linked to the coordinates of neurons."""
    threshold = config.analysis.sequence.spike_threshold if threshold is None else threshold
    # TODO: load_coordinates_and_rate to load_rate
    coordinates, rate = PIC.load_coordinates_and_rate(config, tag)
    bin_rate = UNI.binarize_rate(rate.T, threshold)
    return extract_spikes(bin_rate, coordinates)


# TODO: either put in class or somewhere else?
# TODO: Add tests!
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
