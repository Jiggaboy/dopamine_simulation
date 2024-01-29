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

import lib.universal as UNI

#===============================================================================
# CLASS
#===============================================================================

class AnalysisParams:

    def __init__(self, config:object):
        self.sequence = SequencesParams()
        self.dbscan_controls = DBscanControls(config)


class SequencesParams:
    spike_threshold = 0.35
    eps = 4.
    # eps = 4.5
    min_samples = 20

    radius = 2


class DBscanControls:
    def __init__(self, config:object):
        # TODO: dataclass?
        self.config = config
        if config.landscape is not None:
            self.detection_spots = self._get_detection_spots(config.landscape.params["size"], config.landscape.params["base"])


    def detection_spots_by_tag(self, tag:str) -> tuple:
        name = UNI.name_from_tag(tag)
        return {key: center for key, center in self.detection_spots}[name]


    @staticmethod
    def _get_detection_spots(perlin_size:int, perlin_base:int):
        detection_spots = []
        # SIZE 4, BASE 1
        if perlin_size == 4 and perlin_base == 1:
            center_in_4_1 = ((76, 40), (56, 59), )
            center_repeater_4_1 = ((24, 36), (5, 40), (47, 55))
            center_activator_4_1 = ((76, 25), (58, 40), (56, 59)) # pre, activated, merged
            center_starter_4_1 = ((55, 5), (48, 16), (34, 22))

            # UNI.append_spot(detection_spots, "in", center_in_4_1)
            # UNI.append_spot(detection_spots, "repeater", center_repeater_4_1)
            UNI.append_spot(detection_spots, "out-activator", center_activator_4_1)
            # UNI.append_spot(detection_spots, "starter", center_starter_4_1)

        else:
            logger.info("No spots defined: No set is used.")
        return detection_spots



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    from params import BaseConfig
    main()
    AnalysisParams(BaseConfig())
