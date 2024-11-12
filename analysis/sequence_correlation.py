#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
    Determines the correlation between two detected sequence series.
    The idea is to relate the effect of a patch to the correlation of sequences.

Description:

Possible extensions:

    # Plots which sequence_id crosses which location (enables manual inspection of the correlations.)
    from plot.sequences import scatter_sequence_at_location
    scatter_sequence_at_location(sequence_at_center, sequence.center)


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


from dataclasses import dataclass, field
import numpy as np


from analysis import DBScan_Sequences
from lib import pickler as PIC
import lib.dopamine as DOP
from lib import universal as UNI

from lib.decorator import functimer

BS_TAG = "bs"
PATCH_TAG = "patch"


#===============================================================================
# CLASS
#===============================================================================
@dataclass
class SequenceCorrelator(DBScan_Sequences):

    # @functimer
    def detect_sequence_at_center(self, tag:str, center:tuple, force:bool=False) -> None:
        """Detects which sequences cross which centers given the spiking data."""

        if not force:
            logger.info("Load Sequences at center.")
            sequence_at_center = PIC.load_sequence_at_center(tag, center, self._config)
            if sequence_at_center is not None:
                return sequence_at_center

        coordinates = UNI.get_coordinates(self._config.rows)

        logger.info("Load spike train and labels")
        spikes, labels = self._scan_spike_train(tag)

        logger.info("Identify which sequence IDs are at the locations.")
        sequence_at_center = self.get_sequences_id_at_location(labels, spikes, center, coordinates)

        logger.info("Save sequences at center.")
        PIC.save_sequence_at_center(sequence_at_center, tag, center, self._config)
        return sequence_at_center


    def count_shared_sequences(self, tag:str, force_patch:bool=False, force_baseline:bool=False) -> None:
        # Attributes are *_labels (1D), *_spikes (3D), *_times (1D for each location)
        center = self._config.analysis.dbscan_controls.detection_spots_by_tag(tag)
        no_of_center = len(center)

        baseline_tag = self._config.get_baseline_tag_from_tag(tag)
        sequence_at_center = self.detect_sequence_at_center(baseline_tag, center, force=force_baseline)
        # corr_baseline = self.calculate_shared_cluster(sequence_at_center, no_of_center)

        sequence_at_center_patch = self.detect_sequence_at_center(tag, center, force=force_patch)


    def get_sequences_id_at_location(self, labels:np.ndarray, spikes:np.ndarray, centers:tuple, coordinates:np.ndarray) -> np.ndarray:
        """
        Identifies which sequence IDs cross the certain locations.

        Parameters
        ----------
        labels : np.ndarray
            Cluster labels.
        spikes : np.ndarray
            Correspnding cluster data (time, x, y).
        centers : tuple
            Iterable of (x, y) positions. Spike coordinates are compared to these locations.
        coordinates : np.ndarray
            All possible x- and y-coordinates of the network.

        Returns
        -------
        sequence_at_center : np.ndarray
            Shape: (sequence id, center). Boolean array whether a sequence crossed a center location.

        """

        label_identifier = sorted(set(labels))

        sequence_at_center = np.zeros((len(label_identifier), len(centers)), dtype=bool)

        neuron_coordinates_at_centers = [
            DOP.circular_patch(self._config.rows, center, self._config.analysis.sequence.radius)
            for center in centers]

        for label in label_identifier[:]:
            # Labels is a numpy array of the clustered data.
            # label an identifier (int)
            label_idx = labels == label
            # Find those spikes which correspond to that sequence
            spikes_in_sequence = spikes[label_idx]
            # Find those labels, which cross a location
            for c, center in enumerate(centers):
                sequence_at_center[label, c] = self.has_spikes_at_center(spikes_in_sequence, coordinates[neuron_coordinates_at_centers[c]])
        return sequence_at_center


    def has_spikes_at_center(self, spikes_in_sequence:np.ndarray, coordinates:np.ndarray) -> bool:
        """
        Detects whether a the spikes crossed a location at any point.

        Parameters
        ----------
        spikes_in_sequence : np.ndarray
            Spike train. First column is the time, 2nd and 3rd the x- and y-coordinates respectively.
        coordinates : np.ndarray
            The set of all coordinates in the system.

        Returns
        -------
        bool

        """
        # Checks whether any spikes-information shares (all) the xy-coordinates with the neurons at that location.
        idx = (spikes_in_sequence[:, 1:][:, np.newaxis] == coordinates).all(-1).any(-1)
        return np.count_nonzero(idx)
