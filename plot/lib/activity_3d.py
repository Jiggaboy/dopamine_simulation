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

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """"""
    pass


#===============================================================================
# METHODS
#===============================================================================


def plot_cluster(data:np.ndarray, labels:np.ndarray=None, force_label:int=None, title=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("time")
    ax.set_ylabel("X-Position")
    ax.set_zlabel("Y-Position")
    ax.set_title(title)

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



def plot_sequences_at_locations(sequence_at_center:np.ndarray, sequence:object, is_baseline:bool=True)->None:
    """
    Test the overall distribution of spikes in sequences (identified the issue with the shift).
    """
    for c in range(len(sequence.center)):
        # Find labels that cross that center (by index)
        labels_that_cross = sequence_at_center[:, c].nonzero()[0]
        # Get index of those which share the
        labels = sequence.baseline_labels if is_baseline else sequence.patch_labels
        spikes_in_seqs = np.isin(labels, labels_that_cross)
        title_tag = "Baseline" if is_baseline else "Patch"
        title = f"{title_tag} - Center: {sequence.center[c]}"
        plot_cluster(sequence.baseline_spikes[spikes_in_seqs], sequence.baseline_labels[spikes_in_seqs], title=title)










if __name__ == '__main__':
    main()
