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
import numpy as np
import matplotlib.pyplot as plt


from lib import pickler as PIC
import lib.dopamine as DOP
from lib import universal as UNI
from params import config


from plot.sequences import _plot_cluster

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""
    all_tags = config.get_all_tags("repeat-early")
    correlator = SequenceCorrelator(config)
    for tag in all_tags:
        print(tag)
        correlator.count_shared_sequences(tag)
        # break
    plt.show()


#===============================================================================
# CLASS
#===============================================================================

@dataclass
class SequenceCorrelator:
    config: object


    def has_spikes_at_center(self, spikes_in_sequence:np.ndarray, center:tuple, coordinates:np.ndarray) -> bool:
        """
        Detects whether a the spikes crossed a circular location at any point.

        Parameters
        ----------
        spikes_in_sequence : np.ndarray
            Spike train. First column is the time, 2nd and 3rd the x- and y-coordinates respectively.
        center : tuple
            XY-Position.
        coordinates : np.ndarray
            The set of all coordinates in the system.

        Returns
        -------
        bool

        """
        neuron_id = DOP.circular_patch(self.config.rows, center, self.config.analysis.sequence.radius)
        neuron_coordinates = coordinates[neuron_id]
        # Checks whether any spikes-information shares (all) the xy-coordinates with the neurons at that location.
        idx = (spikes_in_sequence[:, 1:][:, np.newaxis] == neuron_coordinates).all(-1).any(-1)
        return np.count_nonzero(idx)


    def get_sequences_id_at_location(self, labels:np.ndarray, spikes:np.ndarray, centers:tuple, coordinates:np.ndarray):
        label_identifier = sorted(set(labels))

        sequence_at_center = np.zeros((len(label_identifier), len(centers)), dtype=bool)

        for label in label_identifier[:]:
            label_idx = labels == label
            spikes_in_sequence = spikes[label_idx]

            # Find those labels, which cross a location
            for c, center in enumerate(centers):
                sequence_at_center[label, c] = self.has_spikes_at_center(spikes_in_sequence, center, coordinates)
        return sequence_at_center



    def count_shared_sequences(self, tag:str):
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
        # for c in range(len(sequence.center)):
        #     labels_that_cross = sequence_at_center[:, c].nonzero()[0]
        #     spikes_in_seqs = np.isin(sequence.baseline_labels, labels_that_cross)
        #     title = f"Baseline - Center: {sequence.center[c]}"
        #     _plot_cluster(sequence.baseline_spikes[spikes_in_seqs], sequence.baseline_labels[spikes_in_seqs], title=title)

        # for c in range(len(sequence.center)):
        #     labels_that_cross = sequence_at_center_patch[:, c].nonzero()[0]
        #     spikes_in_seqs = np.isin(sequence.patch_labels, labels_that_cross)
        #     title = f"Patch - Center: {sequence.center[c]}"
        #     _plot_cluster(sequence.patch_spikes[spikes_in_seqs], sequence.patch_labels[spikes_in_seqs], title=title)


        # spikes_in_seqs = np.isin(sequence.baseline_labels, [57])
        # _plot_cluster(sequence.baseline_spikes[spikes_in_seqs], sequence.baseline_labels[spikes_in_seqs])

        no_of_center = len(sequence.center)
        def calculate_shared_cluster(sequence_at_center, no_of_center:int):
            corr = np.zeros(shape=(no_of_center, no_of_center))
            for c in range(no_of_center):
                for r in range(no_of_center):
                    labels_at_c = sequence_at_center[:, c].nonzero()[0]
                    labels_at_r = sequence_at_center[:, r].nonzero()[0]

                    also_clustered_at_other_center = np.isin(labels_at_c, labels_at_r)
                    shared = also_clustered_at_other_center.nonzero()[0].size / labels_at_r.size
                    corr[c, r] = shared
                    print(r, c, shared)
            return corr

        corr_baseline = calculate_shared_cluster(sequence_at_center, no_of_center)
        corr_patch = calculate_shared_cluster(sequence_at_center_patch, no_of_center)

        plt.figure()
        plt.title("Baseline")
        im = plt.imshow(corr_baseline)
        plt.colorbar(im)
        plt.figure()
        plt.title("Patch")
        im = plt.imshow(corr_patch)
        plt.colorbar(im)

        # test = sequence_at_center * np.arange(1, len(sequence.center)+1)
        # test_patch = sequence_at_center_patch * np.arange(1, len(sequence.center)+1)
        # plt.figure()
        # plt.plot(test, marker="*", ls="None")
        # plt.figure()
        # plt.plot(test_patch, marker="*", ls="None")
        # plt.show()

        # plot ths histogram for each neuron the distance between spikes in time
        # Filter?


if __name__ == '__main__':
    main()
