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
import matplotlib.pyplot as plt


from analysis import DBScan_Sequences
from lib import pickler as PIC
import lib.dopamine as DOP
from lib import universal as UNI
from params import config


from plot.lib import plot_cluster
from plot.sequences import imshow_correlations, imshow_correlation_difference
from lib.decorator import functimer

BS_TAG = "bs"
PATCH_TAG = "patch"

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    all_tags = config.get_all_tags("repeat-early")
    correlator = SequenceCorrelator(config)
    force_patch = UNI.yes_no("Force clustering for patch simulations?")
    force_baseline = UNI.yes_no("Force baseline clustering?")
    for tag in all_tags:
        correlator.count_shared_sequences(tag, force_patch=force_patch, force_baseline=force_baseline)
        # imshow_correlations(correlator.correlations[tag][BS_TAG], tag=tag)
        # imshow_correlations(correlator.correlations[tag][PATCH_TAG], is_baseline=False, tag=tag)
    #     break
    # plt.show()
    # return

    for tag_across_seeds in config.get_all_tags("repeat-early", seeds="all"):
        fig, axes = plt.subplots(ncols=len(tag_across_seeds) + 1, nrows=3)
        for t, tag in enumerate(tag_across_seeds):
            imshow_correlations(correlator.correlations[tag][BS_TAG], tag=tag, ax=axes[0, t])
            imshow_correlations(correlator.correlations[tag][PATCH_TAG], is_baseline=False, tag=tag, ax=axes[1, t])
            corr_diff = correlator.correlations[tag][PATCH_TAG] - correlator.correlations[tag][BS_TAG]
            imshow_correlation_difference(corr_diff, ax=axes[2, t])

        # Plot averages
        correlation_avgs = {}
        for i, id_ in enumerate((BS_TAG, PATCH_TAG)):
            corr = [correlator.correlations[tag][id_] for tag in tag_across_seeds]
            correlation_avgs[id_] = np.asarray(corr).mean(axis=0)

        for i, id_ in enumerate((BS_TAG, PATCH_TAG)):
            imshow_correlations(correlation_avgs[id_], tag="Average", ax=axes[i, -1], is_baseline=bool(i))
        corr_diff = correlation_avgs[PATCH_TAG] - correlation_avgs[BS_TAG]
        imshow_correlation_difference(corr_diff, ax=axes[2, -1])


        # break
    plt.show()


#===============================================================================
# CLASS
#===============================================================================
@dataclass
class SequenceCorrelator(DBScan_Sequences):
    correlations: None = field(default_factory=dict)


    @functimer
    def detect_sequence_at_center(self, tag:str, center:tuple, force:bool=False) -> None:

        if not force:
            sequence_at_center = PIC.load_sequence_at_center(tag, center)
            if sequence_at_center is not None:
                return sequence_at_center

        coordinates = UNI.get_coordinates(self._config.rows)

        logger.info("Load spike train and labels")
        spikes, labels = self._scan_spike_train(tag)

        sequence_at_center = self.get_sequences_id_at_location(labels, spikes, center, coordinates)
        PIC.save_sequence_at_center(sequence_at_center, tag, center)
        return sequence_at_center


    @functimer
    def count_shared_sequences(self, tag:str, force_patch:bool=False, force_baseline:bool=False) -> None:
        # Attributes are *_labels (1D), *_spikes (3D), *_times (1D for each location)
        center = self._config.analysis.dbscan_controls.detection_spots_by_tag(tag)
        no_of_center = len(center)

        baseline_tag = self._config.get_baseline_tag_from_tag(tag)
        sequence_at_center = self.detect_sequence_at_center(baseline_tag, center, force=force_baseline)
        corr_baseline = self.calculate_shared_cluster(sequence_at_center, no_of_center)

        sequence_at_center_patch = self.detect_sequence_at_center(tag, center, force=force_patch)
        corr_patch = self.calculate_shared_cluster(sequence_at_center_patch, no_of_center)

        self.correlations[tag] = {BS_TAG: corr_baseline, PATCH_TAG: corr_patch}

        # from plot.sequences import scatter_sequence_at_location
        # scatter_sequence_at_location(sequence_at_center, sequence.center)
        # scatter_sequence_at_location(sequence_at_center_patch, sequence.center)
        # [print(key, self.correlations[tag][key]) for key in self.correlations[tag].keys()]


    def has_spikes_at_center(self, spikes_in_sequence:np.ndarray, coordinates:np.ndarray) -> bool:
        """
        Detects whether a the spikes crossed a circular location at any point.

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


    def get_sequences_id_at_location(self, labels:np.ndarray, spikes:np.ndarray, centers:tuple, coordinates:np.ndarray):
        label_identifier = sorted(set(labels))

        sequence_at_center = np.zeros((len(label_identifier), len(centers)), dtype=bool)

        neuron_coordinates_at_centers = [DOP.circular_patch(self._config.rows, center, self._config.analysis.sequence.radius)
                                         for center in centers]

        for label in label_identifier[:]:
            label_idx = labels == label
            spikes_in_sequence = spikes[label_idx]
            # Find those labels, which cross a location
            for c, center in enumerate(centers):
                sequence_at_center[label, c] = self.has_spikes_at_center(spikes_in_sequence, coordinates[neuron_coordinates_at_centers[c]])
        return sequence_at_center



    @staticmethod
    def calculate_shared_cluster(sequence_at_center, no_of_center:int):
        corr = np.zeros(shape=(no_of_center, no_of_center))
        for c in range(no_of_center):
            for r in range(no_of_center):
                labels_at_c = np.flatnonzero(sequence_at_center[:, c])
                labels_at_r = np.flatnonzero(sequence_at_center[:, r])

                also_clustered_at_other_center = np.isin(labels_at_c, labels_at_r)
                shared = np.count_nonzero(also_clustered_at_other_center) / labels_at_r.size
                corr[c, r] = shared
        return corr


if __name__ == '__main__':
    main()
