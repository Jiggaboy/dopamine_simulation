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

from plot.constants import KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE
from plot.constants import COLOR_MAP_ACTIVITY, COLOR_MAP_DIFFERENCE, NORM_DIFFERENCE, NORM_ACTIVITY
from lib.universal  import dotdict


#===============================================================================
# CLASSES
# Each class configures a certain figure.
#===============================================================================

class AnimationConfig:
    save_animations = False

    animation_kwargs = dotdict({
        "start": 0,
        "stop": None,
        "interval": 1000 / 20, # 1000 / x -> x frames per second
        "step": 8,             # steps across the index (in time)
        "add_spikes": False,
    })

    difference_frame = {
        "num": "baseline_differences",
        "figsize": (3.5, 3.5),
    }

    figure_frame = {
        "figsize": (5, 3.5),
    }

    image = {
        "norm": NORM_ACTIVITY,
        "cmap": COLOR_MAP_ACTIVITY,
    }

    difference_image = {
        "norm": NORM_DIFFERENCE,
        "cmap": COLOR_MAP_DIFFERENCE,
    }
