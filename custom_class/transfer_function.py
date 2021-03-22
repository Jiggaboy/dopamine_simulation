#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:54:00 2021

@author: hauke
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x:float, factor:float=1.0, x0:float=0.0, steepness:float=1.0)->float:
    return factor / (1.0 + np.exp(steepness*(x0 - x)))


def transfer_function(input_:float)->float:
    return sigmoid(input_,  x0=21, steepness=.2) # symmetric


x0 = 40
x = np.arange(0, x0)
plt.plot(x, transfer_function(x))
