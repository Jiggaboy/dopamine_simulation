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
from params import config


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""
    all_tags = config.get_all_tags("repeater")
    correlator = SequenceCorrelator(config)
    for tag in all_tags:
        print(tag)
        correlator.count_shared_sequences(tag)
        # break
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


if __name__ == '__main__':
    main()
