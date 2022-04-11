#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:54:00 2021

@author: hauke
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TransferFunction:
    offset: float
    slope: float

    def run(self, input_: float):
        return sigmoid(input_, x0=self.offset, steepness=self.slope)


def sigmoid(x:float, factor:float=1.0, x0:float=0.0, steepness:float=1.0)->float:
    return factor / (1.0 + np.exp(steepness*(x0 - x)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dx = 40
    offset = 10
    slope = .1
    t = TransferFunction(offset, slope)
    x = np.arange(offset-dx, offset+dx)
    plt.plot(x, t.run(x))
    plt.xlabel("synaptic input + external drive")
    plt.ylabel("output rate")
    plt.title("(sigmoidal) Gain function")
