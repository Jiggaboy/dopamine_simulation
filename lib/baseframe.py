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
from lib import experiment_info, functimer

#===============================================================================
# CLASS
#===============================================================================


class BaseFrame:

    def __init__(self):
        self.__post_init__()


    def __post_init__(self):
        _, _drugs = experiment_info.get_data_dir()
        self._drugs = sorted(_drugs)



#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    main()
