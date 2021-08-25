#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:54:00 2021

@author: hauke
"""
import numpy as np
import matplotlib.pyplot as plt


import configuration as CF

def sigmoid(x:float, factor:float=1.0, x0:float=0.0, steepness:float=1.0)->float:
    return factor / (1.0 + np.exp(steepness*(x0 - x)))


def transfer_function(input_:float)->float:
    return sigmoid(input_,  x0=CF.X_OFFSET, steepness=CF.STEEPNESS) # symmetric


if __name__ == '__main__':
    dx = 40
    x = np.arange(CF.X_OFFSET-dx, CF.X_OFFSET+dx)
    plt.plot(x, transfer_function(x))
    plt.xlabel("synaptic input + external drive")
    plt.ylabel("output rate")
    plt.title("(sigmoidal) Gain function")
