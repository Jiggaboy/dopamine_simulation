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
import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

from plot.constants import KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE
from plot.constants import COLOR_MAP_ACTIVITY, COLOR_MAP_DIFFERENCE, NORM_DIFFERENCE, NORM_ACTIVITY
from lib.universal import dotdict

#===============================================================================
# CLASSES
# Each class configures a certain figure.
#===============================================================================

class AnimationConfig:
    save_animations = False

    animation_kwargs = dotdict({
        "start": 200,
        "stop": None,
        "interval": 1000 / 25, # 1000 / x -> x frames per second
        "step": 2,             # steps across the index (in time)
    })

    difference_frame = {
        "num": "baseline_differences",
        "figsize": (8, 6),
    }

    figure_frame = {
        "figsize": (4, 3.4),
    }

    image = {
        "norm": NORM_ACTIVITY,
        "cmap": COLOR_MAP_ACTIVITY,
    }

    difference_image = {
        "norm": NORM_DIFFERENCE,
        "cmap": COLOR_MAP_DIFFERENCE,
    }





class ConnectivityDistributionConfig:
    FIG_PARAMS = {
        "num": "joint_connectivity",
        "figsize": (3.4, 3)
    }

    NROWS = 60
    # Targets
    targets = {
        "center": np.asarray((30, 30), dtype=int),
        "std": 4,
        "n_conn": 300,
        "marker": ".",
        "ms": 4,
        "linestyle": "None",

    }
    CENTER = np.asarray((30, 30), dtype=int)
    STD = 4
    SHIFT = 5
    N_CONN = 300
    # Targets (style)
    MARKER = "."
    C_TARGET = KTH_GREEN
    C_TARGET_SHIFTED = KTH_BLUE
    NEURON_SIZE = 6
    C_NEURON = "red"


    #### STYLE HISTOGRAMS
    C_INH_HIST = KTH_PINK
    C_FULL_HIST = KTH_GREY
    LW_DIST = 2

    MAX_HIST = 12
    BIN_WIDTH = 1

    #### SCALEBAR
    X_SCALEBAR = 14
    Y_SCALEBAR = 55
    WIDTH_SCALEBAR = 2


class ActivityDifferenceConfig:
    figure_frame = {
        "figsize": (4, 3),
    }

    font = {
        "fontsize": 20,
    }

    image = {
        "norm": (-.3, .3),
        "cmap": plt.cm.seismic,
    }




#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    """Description of main()"""
    pass




if __name__ == '__main__':
    main()
