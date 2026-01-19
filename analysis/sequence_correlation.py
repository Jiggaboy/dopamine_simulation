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
__version__ = '0.2'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger


from dataclasses import dataclass
import numpy as np

import pandas as pd
import sklearn.cluster as cluster
from collections import OrderedDict

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

    def count_shared_sequences(self, tag:str, force_patch:bool=False, force_baseline:bool=False) -> None:
        # Attributes are *_labels (1D), *_spikes (3D), *_times (1D for each location)
        detection_spots = self._config.analysis.dbscan_controls.detection_spots_by_tag(tag)

        baseline_tag = self._config.get_baseline_tag_from_tag(tag)
        self.detect_sequence_at_center(baseline_tag, detection_spots, force=force_baseline)
        self.detect_sequence_at_center(tag, detection_spots, force=force_patch)

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
        # Save also the labels/identities of the sequences.
        PIC.save_sequence_at_center(sequence_at_center, tag, center, self._config)
        return sequence_at_center



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
            sequence_id_key = "sequence id"
            sequence_at_center = pd.DataFrame(columns=[sequence_id_key])

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

                # Update in v0.2
                db_center = cluster.DBSCAN(min_samples=1, eps=5)
                sequence_cross_center = OrderedDict({})
                for c, center in enumerate(centers):
                    sequence_cross_center[f"C{c}"] = [] # Runs (usually) from C0 to C2

                    # Find those spikes that are crossing the coordinates of the detection spot
                    spikes_at_coordinate = (spikes_in_sequence[:, 1:][:, np.newaxis] == coordinates[neuron_coordinates_at_centers[c]]).all(-1) # Shape is (spikes, neurons_at_center)
                    # spikes_at_coordinate.any(-1) # Shape is (Spikes); Array is boolean and indicates that the spikes was detected at the detection spot.
                    # Retrieve the index of these spikes
                    idx = np.argwhere(spikes_at_coordinate.any(-1)).flatten()
                    if not idx.any():
                        continue # Actually set the indication?
                    # Extract times at which the spots is crossed
                    spike_timing_at_coordinates = spikes_in_sequence[idx][:, 0] # Get all the time points
                    db_center.fit(spike_timing_at_coordinates.reshape(-1, 1))
                    assert (db_center.labels_ >= 0).all()


                    for l in set(db_center.labels_):
                        mean_spike_timing_at_coordinates = spike_timing_at_coordinates[db_center.labels_ == l].mean()
                        sequence_cross_center[f"C{c}"].append(mean_spike_timing_at_coordinates)

                all_mean_spike_times = []
                for times in sequence_cross_center.values():
                    if not times:
                        continue
                    all_mean_spike_times.extend(times)
                all_mean_spike_times = np.asarray(all_mean_spike_times)
                if not all_mean_spike_times.size:
                    continue # Actually set the indication
                db_spots = cluster.DBSCAN(min_samples=1, eps=200) # 500
                db_spots.fit(all_mean_spike_times.reshape(-1, 1))

                indicator = np.zeros((len(centers), db_spots.labels_.max()+1), dtype=bool)
                indicator_time = np.zeros((len(centers), db_spots.labels_.max()+1), dtype=float)
                row = {
                    sequence_id_key: np.full(indicator.shape[1], fill_value=label, dtype=int),
                }
                pre = 0
                for c in range(len(centers)):
                    idx = slice(pre, pre+len(sequence_cross_center[f"C{c}"]))
                    indicator[c, db_spots.labels_[idx]] = True
                    indicator_time[c, db_spots.labels_[idx]] = sequence_cross_center[f"C{c}"]
                    pre += len(sequence_cross_center[f"C{c}"])
                    row[f"C{c}"] = indicator[c]
                    row[f"C{c}_time"] = indicator_time[c]

                row_df = pd.DataFrame.from_dict(row)
                sequence_at_center = pd.concat([sequence_at_center, row_df], ignore_index=True)
            return sequence_at_center


def has_spikes_at_center(spikes_in_sequence:np.ndarray, coordinates:np.ndarray) -> bool:
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
