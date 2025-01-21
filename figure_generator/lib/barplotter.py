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
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataclasses import dataclass

from lib import pickler as PIC



x_bs = 0.
x_patch = 1.

#===============================================================================
# METHODS
#===============================================================================

def bar(order, sequences, tag, width=.4) -> object:

    name = tag
    # Grab the figure and axes
    if not plt.fignum_exists(name):
        fig, ax = plt.subplots(num=name, layout='constrained', figsize=(3, 3))
        offset = width
        ax.set_xticks(
            np.arange(len(sequences)), order,
            rotation=0,
        )
        ax.set_yticks([0., 5., 10., 15., 20.],)
        ax.set_ylim(0, 24)
        ax.set_ylabel('sequence count')
        ax.set_title('Sequence count across detection spots')
        label = "baseline"
    else:
        fig = plt.figure(name)
        ax = fig.axes
        offset = 0
        label = "patch"

    avg = np.asarray([i.mean() for i in sequences.values()])
    plt.bar(x=np.arange(len(sequences))-offset, height=avg, width=width, align="edge", label=label)
    return fig


def reorder(shared:OrderedDict, order:list) -> OrderedDict:
    """
    Rearranges an OrderedDict according to another order which contains the same keys.
    Only keys in order are found in the returned OrderedDict.
    """
    ordered = OrderedDict()
    for o in order:
        ordered[o] = shared[o]
    return ordered

#===============================================================================
# CLASS
#===============================================================================

@dataclass
class BarPlotter:
    config:object
    tags: list
    labels: list
    detection_spots: list

    def get_sequences_across_seeds(self, keys:list, is_baseline:bool=False) -> dict:
        shared_all_seeds = OrderedDict({l: np.zeros(len(self.tags)) for l in self.labels})
        for idx, tag in enumerate(self.tags):
            _tag = tag if not is_baseline else self.config.get_baseline_tag_from_tag(tag)
            sequence_at_center = PIC.load_sequence_at_center(_tag, self.detection_spots, self.config)
            shared = self._create_shared(sequence_at_center, self.detection_spots)

            for label, seq in zip(self.labels, keys):
                if isinstance(shared[seq], (int, float)):
                    shared_all_seeds[label][idx] = shared[seq]
                elif isinstance(shared[seq], np.ndarray):
                    shared_all_seeds[label][idx] = shared[seq].size
                else:
                    raise TypeError("No valid type given.")
        return shared_all_seeds


    def bar_sequences(self, sequences_across_seeds:dict, axes:list, is_baseline:bool=False) -> None:
        x_pos = x_bs if is_baseline else x_patch
        for idx, ax in enumerate(axes[:-1]):
            shared_within_seed = {key: value[idx] for key, value in sequences_across_seeds.items()}
            self._bar_shared(x_pos, shared_within_seed, axis=ax, average=False)
        add_labels = True if is_baseline else False
        self._bar_shared(x_pos, sequences_across_seeds, axis=axes[-1], average=True, add_labels=add_labels)


    def _bar_shared(self, x, shared, axis:object=None, average:bool=False, add_labels:bool=False):
        if axis is None:
            plt.gca().set_prop_cycle(None)
        else:
            axis.set_prop_cycle(None)

        ax = axis if axis is not None else plt
        y0 = 0

        for key, value in shared.items():
            height = value.mean() if average else value
            label = key if add_labels else None
            bar, y0 = self._stacked_bar(x, height, y0, label=label, axis=ax)

            # Annotation
            if height > 1.:
                text = int(height) if not average else height
                xy = x, y0 - height / 2
                ann_kwargs = {
                    "ha": "center",
                    "va": "center",
                }
                ax.annotate(text, xy, **ann_kwargs)


    def _create_shared(self, sequence_at_center:np.ndarray, detection_spots:np.ndarray) -> dict:
        if len(detection_spots) == 2:
            return self._create_shared_two_spots(sequence_at_center, detection_spots)

        shared = OrderedDict({})
        for i in range(len(detection_spots)):
            shared[str(i)] = np.count_nonzero(sequence_at_center[:, i])
            mask = np.zeros(len(detection_spots), dtype=bool)
            mask[i] = True
            shared["not " + str(i)] = self._get_shared(sequence_at_center, np.invert(mask))


        shared["all"] = (sequence_at_center[:, 0]) & \
                        (sequence_at_center[:, 1]) & \
                        (sequence_at_center[:, 2])
        shared["all"] = np.count_nonzero(shared["all"])
        return shared


    ##### Counts sequences iff the spot (and not any other) is active
    ##### Works for 3 spots
    # def _create_shared(self, sequence_at_center:np.ndarray, detection_spots:np.ndarray) -> dict:

    #     shared = OrderedDict({})
    #     for i in range(len(detection_spots)):
    #         mask = np.zeros(len(detection_spots), dtype=bool)
    #         mask[i] = True

    #         shared[str(i)] = self._get_shared(sequence_at_center, mask)
    #         shared["not " + str(i)] = self._get_shared(sequence_at_center, np.invert(mask))

    #     all_shared = np.count_nonzero(sequence_at_center.all(axis=1))
    #     shared["all"] = all_shared
    #     return shared

    ##### Respects only whether a sequence crosses a particular spot
    ##### Only works for 2 spots
    def _create_shared_two_spots(self, sequence_at_center:np.ndarray, detection_spots:np.ndarray) -> dict:

        shared = OrderedDict({})
        for i in range(len(detection_spots)):
            shared[str(i)] = np.where(sequence_at_center[:, i] == True)[0]
            shared["not " + str(i)] = np.where(sequence_at_center[:, ~i] == True)[0]
        all_shared = sequence_at_center.all(axis=1).nonzero()
        shared["all"] = all_shared[0]
        return shared


    @staticmethod
    def _stacked_bar(x, y, y0=0., axis:object=None, **bar_kwargs):
        return axis.bar(x, y, bottom=y0, **bar_kwargs), y + y0


    @staticmethod
    def _get_shared(sequence_at_center:np.ndarray, mask:np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask).ravel()
        return np.count_nonzero((sequence_at_center[:, idx]).all(axis=-1))


    @staticmethod
    def init_axes(axes:list) -> None:
        axes[0].set_ylabel("Sequence count")
        axes[-1].set_title("Averaged across seeds")

        for ax in axes:
            ax.set_xticks([x_bs, x_patch])
            ax.set_xticklabels(["baseline", "with patch"], rotation=30)
        for idx, ax in enumerate(axes[:-1]):
            ax.set_title(f"Seed: {idx}")





#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    main()
