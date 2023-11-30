#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:04:00 2021

@author: hauke
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("##################################################################### RATE imported #####################################################################")
def rate(rates:np.ndarray, avg:bool=False, threshold:float=None, figname:str=None):
    plt.figure(figname)
    style = {}
    style_avg = {"linewidth": 5,
                 "color": "red",
                 "label": "avg"
                 }
    style_thr = {"color": "black",
                 "ls": "dotted",
                 "label": "threshold"
                 }

    if avg:
        style["ls"] = "dotted"
        # labels.insert(1, "avg")

    plt.plot(rates.T, **style)
    if avg:
        handle_avg = plt.plot(rates.mean(axis=0), **style_avg)

    if threshold:
        handle_thr = plt.axhline(threshold, **style_thr)

    plt.xlabel("time in ms")
    plt.ylabel("activation")
    plt.legend()
    plt.title("Rates")
