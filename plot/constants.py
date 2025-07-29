#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""
import matplotlib.pyplot as plt
import numpy as np

COLOR_MAP_ACTIVITY = plt.cm.gist_heat_r
COLOR_MAP_DIFFERENCE = plt.cm.seismic

NORM_DIFFERENCE = -.25, .25
NORM_ACTIVITY = 0, 1

############ COLORS #########################

KTH_GREEN = np.asarray((176, 201, 43)) / 255
KTH_PINK = np.asarray((216, 84, 151)) / 255
KTH_GREY = np.asarray((101, 101, 108)) / 255
KTH_BLUE = np.asarray((25, 84, 166)) / 255
KTH_LIGHT_BLUE = np.asarray((36, 160, 216)) / 255

## Updated on Jan 24. 2025
KTH_GREEN = np.asarray((77, 160, 97)) / 255
KTH_PINK = np.asarray((232, 106, 88)) / 255
KTH_GREY = np.asarray((50, 50, 50)) / 255
KTH_BLUE = np.asarray((0, 71, 145)) / 255
KTH_LIGHT_BLUE = np.asarray((98, 152, 210)) / 255
KTH_YELLOW = np.asarray((166, 89, 0)) / 255
KTH_YELLOW = np.asarray((255, 190, 0)) / 255

COLORS = (KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE, KTH_LIGHT_BLUE, KTH_YELLOW)
