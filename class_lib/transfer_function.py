#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""


import numpy as np

import sys
print(sys.path)


class TransferFunction:

    def __init__(self, offset, slope):
        self.offset = offset
        self.slope = slope

    def run(self, input_: float):
        return sigmoid(input_, x0=self.offset, steepness=self.slope)


def sigmoid(x:float, factor:float=1.0, x0:float=0.0, steepness:float=1.0)->float:
    return factor / (1.0 + np.exp(steepness*(x0 - x)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dx = 40
    offset = 50
    slope = .1
    t = TransferFunction(offset, slope)
    x = np.arange(offset-dx, offset+dx)
    plt.plot(x, t.run(x))
    plt.xlabel("synaptic input + external drive")
    plt.ylabel("output rate")
    plt.title("(sigmoidal) Gain function")
    plt.show()
