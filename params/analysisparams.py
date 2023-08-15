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

import cflogger
logger = cflogger.getLogger()

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable


import lib.universal as UNI

#===============================================================================
# CLASS
#===============================================================================

class AnalysisParams:

    def __init__(self, config:object):
        self.sequence = SequencesParams(config)
        self.dbscan_controls = DBscanControls(config)
        self.subspaceangels_controls = SubspaceAngleControls()


class SequencesParams:
    spike_threshold = 0.2
    eps = 5
    min_samples = 20
    time_span = 3
    sequence_threshold = 3
    td = 1

    radius = 2

    def __init__(self, config:object):
        self.minimal_peak_distance = config.TAU


class DBscanControls:
    sequences_across_baselines = False
    run_dbscan = True
    sequence_by_cluster = True

    def __init__(self, config:object):
        self.detection_spots = self._get_detection_spots(config.landscape.params["size"], config.landscape.params["base"])

    @staticmethod
    def _get_detection_spots(perlin_size:int, perlin_base:int):
        detection_spots = []
        # SIZE 4, BASE 1
        if perlin_size == 4 and perlin_base == 1:
            center_in_4_1 = ((30, 18), (28, 26), )
            center_linker_4_1 = ((21, 65), (67, 30), (30, 66))
            center_repeater_4_1 = ((9, 37), (2, 32), (55, 49))
            center_activator_4_1 = ((67, 30), (50, 37), (60, 46))
            center_starter_4_1 = ((46, 3), (49, 6), (43, 13))
            center_gate_4_1 = ((28, 24), (28, 39), (20, 32))

            UNI.append_spot(detection_spots, "in", center_in_4_1)
            UNI.append_spot(detection_spots, "linker", center_linker_4_1)
            UNI.append_spot(detection_spots, "repeater", center_repeater_4_1)
            UNI.append_spot(detection_spots, "out-activator", center_activator_4_1)
            UNI.append_spot(detection_spots, "starter", center_starter_4_1)
            UNI.append_spot(detection_spots, "gate-top", center_gate_4_1)
            UNI.append_spot(detection_spots, "gate-bottom", center_gate_4_1)
        # SIZE 4, BASE 10
        elif perlin_size == 4 and perlin_base == 10:
            center_CL_4_10 = ((16, 21), (11, 16), (1, 26))
            center_CL_4_10 = ((58, 12), )#(14, 44), )
            center_TR_4_10= ((56, 58), (64, 62), (0, 1))
            center_TC_4_10 = ((44, 60), (35, 60), (24, 65))
            UNI.append_spot(detection_spots, "starter-CL", center_CL_4_10)
            UNI.append_spot(detection_spots, "starter-TR", center_TR_4_10)
            UNI.append_spot(detection_spots, "starter-TC", center_TC_4_10)
        # SIZE 4, BASE 2
        else:
            logger.info("No spots defined: A default set is used.")
            center_CL_4_10 = ((16, 21), (11, 16), (1, 26))
            center_CL_4_10 = ((58, 12), )#(14, 44), )
            center_TR_4_10= ((56, 58), (64, 62), (0, 1))
            center_TC_4_10 = ((44, 60), (35, 60), (24, 65))
            UNI.append_spot(detection_spots, "starter-CL", center_CL_4_10)
            UNI.append_spot(detection_spots, "starter-TR", center_TR_4_10)
            UNI.append_spot(detection_spots, "starter-TC", center_TC_4_10)
        return detection_spots


class SubspaceAngleControls:
    patch_against_baseline = True
    patch_against_patch = True
    baseline_against_baseline = True


#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    from params import BaseConfig
    main()
    AnalysisParams(BaseConfig())
