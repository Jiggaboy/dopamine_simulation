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
from collections.abc import Iterable

from lib import universal as UNI

#===============================================================================
# METHODS
#===============================================================================


def plot_cluster(data:np.ndarray, labels:np.ndarray=None, force_label:(int, Iterable)=None, title=None):
    print(f"is called with force label {force_label}")
    data_tmp = np.copy(data)

    if not plt.fignum_exists("cluster"):
        fig = plt.figure(num="cluster", figsize=(3, 3))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = plt.figure("cluster")
        ax = fig.axes[0]
    ax.set_xlabel("Time [au]")
    ax.set_ylabel("X-Position") # Reversed (ylabel = x-pos; zlabel=y-pos)- for visualization purposes changed
    ax.set_zlabel("Y-Position")
    ax.set_xticks([1500, 2500])
    ax.set_yticks([0, 30, 60])
    ax.set_zticks([0, 30, 60])
    ax.set_title(title)
    ax.invert_zaxis()

    if labels is None:
        ax.scatter(*data.T, marker=".")
        return

    unique_labels = np.unique(labels)
    print(unique_labels)
    if force_label is not None:
        force_label = UNI.make_iterable(force_label)
    for l in unique_labels:
        if force_label is not None and l not in force_label:
            continue
        ax.scatter(*data_tmp[labels == l][::20].T, label=l, marker=".", s=.1)
    # plt.legend()



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
