#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-12

@author: Hauke Wernecke
"""

import cflogger
logger = cflogger.getLogger()

from dataclasses import dataclass
import numpy as np
import peakutils as putils


import dopamine as DOP
import universal as UNI



@dataclass
class SequenceDetector:
    radius: float
    threshold: float
    minimal_peak_distance: float


    def passing_sequences(self, rate:np.ndarray, center:tuple, rows:int):
        """
        Creates patches with radius around the center to select neurons.
        Neuronal activity is analysed individually and on average.

        return:
            counts, avg_counts
        """
        patches = [DOP.circular_patch(rows, c, self.radius) for c in center]
        neurons = [UNI.patch2idx(p) for p in patches]

        counts = [self.number_of_sequences(rate, n, avg=False) for n in neurons]
        avg_counts = [c.mean() for c in counts]
        return counts, avg_counts


    def number_of_sequences(self, rate:np.ndarray, neuron:(int, iter), avg:bool=False, normalize:bool=False)->int:
        """
        Detect the number of sequences for the given neuron(s) for a given rate. Depending on the paramter threshold and min_dist.

        Parameters
        ----------
        neuron : (int, iterable)
            The neuron(s) to be analyzed.
        avg : bool, optional
            Average across neurons e.g. for a patch of neurons. The default is False.
        normalize: bool, optional
            If True, normalizes the number of sequences to the time.

        Returns
        -------
        int
            DESCRIPTION.

        """
        if isinstance(neuron, int):
            number =  self._number_of_peaks(rate[neuron])
        else:
            if avg:
                number =  self._number_of_peaks(rate[neuron].mean(axis=0))
            else:
                number = np.zeros(len(neuron))
                for idx, n in enumerate(neuron):
                    spike_index, number[idx] = self._number_of_peaks(rate[n])
        if normalize:
            number = number / rate.shape[1]
        return number


    def _number_of_peaks(self, data:np.ndarray)->tuple:
        """Uses the lib-function of putils to detect the position of peaks"""
        # TODO: Why self.threshold - 1.???
        idx = putils.indexes(data, thres=self.threshold - .1, min_dist=self.minimal_peak_distance, thres_abs=True)
        return idx, idx.size
