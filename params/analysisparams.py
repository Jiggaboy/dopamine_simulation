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
    spike_threshold = 0.5
    eps = 4.5
    min_samples = 20
    sequence_threshold = 3

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

        # SIZE 4, BASE 3
        # if perlin_size == 4 and perlin_base == 3:
        #     center_link = ((65, 75), (78, 65), ) #  upper, lower
        #     center_gate_low = ((26, 12), (48, 9), (41, 32), ) # left,right, merged
        #     center_gate_top = ((41, 32), (30, 57), (17, 41), ) # left, right, merged

        #     # UNI.append_spot(detection_spots, "link", center_link)
        #     # UNI.append_spot(detection_spots, "link-double", center_link)

        #     UNI.append_spot(detection_spots, "gate-low-left", center_gate_low)
        #     UNI.append_spot(detection_spots, "gate-low-right", center_gate_low)

        #     UNI.append_spot(detection_spots, "gate-top-left", center_gate_top)
        #     UNI.append_spot(detection_spots, "gate-top-right", center_gate_top)
        # elif perlin_size == 4 and perlin_base == 4:
        #     # SIZE 4, BASE 4
        #     center_starter = ((33, 62), (12, 48), (17, 78)) #  base, left, right

        #     UNI.append_spot(detection_spots, "starter", center_starter)
        # elif perlin_size == 4 and perlin_base == 5:
        #     # SIZE 4, BASE 5
        #     center_gate = ((17, 42), (20, 61), (1, 50)) #  left, right, merged
        #     # center_starter = (58, 60), (55, 73), (56, 2) # pre, post, center
        #     center_starter = (60, 52), (53, 73), (56, 66) # pre, post, center
        #     # center_repeater = (57, 8), (38, 28), (15, 42) # pre, post, reference
        #     center_repeater = (19, 60), (75, 62) # left, pre, post

        #     UNI.append_spot(detection_spots, "gate-left", center_gate)
        #     UNI.append_spot(detection_spots, "starter", center_starter)
        #     UNI.append_spot(detection_spots, "repeat", center_repeater)
        #     UNI.append_spot(detection_spots, "repeat-early", center_repeater)
        # elif perlin_size == 4 and perlin_base == 6:
            # # SIZE 4, BASE 6
            # center_select = ((55, 39), (53, 20), (37, 38)) # base, left, right
            # # center_select_bottom = ((55, 39), (53, 20), (37, 38)) # base, left, right

            # UNI.append_spot(detection_spots, "select", center_select)
            # UNI.append_spot(detection_spots, "select-alt", center_select)
        #     # UNI.append_spot(detection_spots, "select-bottom", center_select_bottom)
        # elif perlin_size == 4 and perlin_base == 9:
        #     # SIZE 4, BASE 9
        #     center_link = ((79, 62), (71, 27), (56, 63)) # main-pre, main-past, path-past

        #     UNI.append_spot(detection_spots, "link", center_link)
            # UNI.append_spot(detection_spots, "link-left", center_link)
            # UNI.append_spot(detection_spots, "link-right", center_link)
        # elif perlin_size == 4 and perlin_base == 22:
        #     # SIZE 4, BASE 22
        #     center_activator = ((32, 59), (17, 3), (15, 50)) # pre, right, activated
        #     UNI.append_spot(detection_spots, "activator", center_activator)
        # elif perlin_size == 4 and perlin_base == 200:
        #     # SIZE 4, BASE 200
        #     loc = ((10, 10), (20, 10)) # pre, right, activated
        #     for i in range(20):
        #         UNI.append_spot(detection_spots, f"loc-{i}", loc)
        #     UNI.append_spot(detection_spots, "repeater", loc)
        #     UNI.append_spot(detection_spots, "activater", loc)
        #     UNI.append_spot(detection_spots, "starter", loc)

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
