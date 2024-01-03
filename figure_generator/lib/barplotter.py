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
                shared_all_seeds[label][idx] = shared[seq].size
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

        shared = OrderedDict({})
        for i in range(len(detection_spots)):
            mask = np.zeros(len(detection_spots), dtype=bool)
            mask[i] = True

            shared[str(i)] = self._get_shared(sequence_at_center, mask)
            shared["not " + str(i)] = self._get_shared(sequence_at_center, np.invert(mask))

        all_shared = sequence_at_center.all(axis=1).nonzero()
        shared["all"] = all_shared[0]
        return shared


    @staticmethod
    def _stacked_bar(x, y, y0=0., axis:object=None, **bar_kwargs):
        return axis.bar(x, y, bottom=y0, **bar_kwargs), y + y0

    @staticmethod
    def _get_shared(sequence_at_center:np.ndarray, mask:np.ndarray) -> np.ndarray:
        seq_at_mask = (sequence_at_center[:, np.newaxis] == mask).all(axis=-1).nonzero()
        return seq_at_mask[0]

    @staticmethod
    def init_axes(axes:list) -> None:
        axes[0].set_ylabel("Sequence count")
        axes[-1].set_title("Averaged across seeds")

        for ax in axes:
            ax.set_xticks([x_bs, x_patch])
            ax.set_xticklabels(["baseline", "with patch"], rotation=45)
        for idx, ax in enumerate(axes[:-1]):
            ax.set_title(f"Seed: {idx}")





#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    main()
