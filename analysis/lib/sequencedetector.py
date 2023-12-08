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


import lib.dopamine as DOP
import lib.universal as UNI



@dataclass
class SequenceDetector:
    radius: float
    threshold: float
    minimal_peak_distance: float


    def _number_of_peaks(self, data:np.ndarray, bin_width:int)->tuple:
        """Uses the lib-function of putils to detect the position of peaks"""
        idx = putils.indexes(data, thres=self.threshold, min_dist=self.minimal_peak_distance, thres_abs=True)
        return idx * bin_width, idx.size
