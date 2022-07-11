#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2022-07-11

@author: Hauke Wernecke
"""

import matplotlib.patches as mpatches

def white_dashed_circle(center:tuple, radius:float)->None:
    circle = mpatches.Circle(center, radius=radius, fc="None", ec="white", linewidth=2, ls="dashed")
    plt.gca().add_artist(circle)