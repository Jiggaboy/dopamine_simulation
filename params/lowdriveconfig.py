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

import numpy as np
from collections import OrderedDict
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

from params import PerlinConfig
from class_lib import Landscape, ExternalDrive, Synapse, TransferFunction

#===============================================================================
# CLASS
#===============================================================================

class LowDriveConfig(PerlinConfig):
    WARMUP = 500 ###############################
    sim_time = 3000
    rows = 70

    ##################### Patches
    center_range = OrderedDict({
        "repeater": (17, 34),
        #"gate-top": (29, 34),
        #"gate-bottom": (28, 26),
    })

    drive = ExternalDrive(10., 40., seeds=np.arange(2))

    PERCENTAGES = .2,

    synapse = Synapse(weight=1., EI_factor=6.5)
    transfer_function = TransferFunction(50., .25)
    drive = ExternalDrive(15., 20., seeds=np.arange(2))
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4, "base": 1}, seed=0)





#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""





if __name__ == '__main__':
    main()
