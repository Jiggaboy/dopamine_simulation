#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1a'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
import lib.universal as UNI

#===============================================================================
# CLASS
#===============================================================================

class AnalysisParams:

    def __init__(self):
        self.sequence = SequencesParams()
        self.dbscan_controls = DBscanControls()


class SequencesParams:
    spike_threshold = 0.4
    eps = 3.
    min_samples = 75

    radius = 2


class DBscanControls:
    def detection_spots_by_tag(self, tag:str) -> tuple:
        name = UNI.name_from_tag(tag)
        return {key: center for key, center in self.detection_spots}[name]
