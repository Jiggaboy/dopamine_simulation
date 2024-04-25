#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-26

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

COLORS = (KTH_GREEN, KTH_PINK, KTH_GREY, KTH_BLUE, KTH_LIGHT_BLUE)
