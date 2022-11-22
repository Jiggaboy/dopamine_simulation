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

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

#===============================================================================
# CLASS
#===============================================================================

class AnalysisParams:

    def __init__(self, config:object):
        self.sequence = SequencesParams(config)



# @dataclass??
class SequencesParams:
    spike_threshold = 0.3
    eps = 5
    min_samples = 20
    td = 1

    radius = 2

    def __init__(self, config:object):
        self.minimal_peak_distance = config.TAU






#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    from params import BaseConfig
    main()
    AnalysisParams(BaseConfig())
