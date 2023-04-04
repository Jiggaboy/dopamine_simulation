#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Generates the figure of the connectivity kernel.

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from collections import namedtuple

# Gaussian parameter
SIGMA_EXC = 5.
SIGMA_INH = 10


plt.figure("gauss", figsize=(5, 3))
gauss_exc = norm(scale = SIGMA_EXC)
gauss_inh = norm(scale = SIGMA_INH)

x0 = 40
x = np.arange(-x0, x0, .1)

#plot the pdfs of these normal distributions
exc_pdf = gauss_exc.pdf(x)
inh_pdf = gauss_inh.pdf(x)
plt.plot(x, exc_pdf, x, inh_pdf, x, exc_pdf-inh_pdf)
plt.legend(["Distribution of exc. neurons (source)",
            "Distribution of inh. neurons (source)",
            "Difference between exc. and inh. distributions"])
plt.title("Distributions of connectivity probabilities")
plt.ylim(ymax=0.13)
plt.xlim(-x0, x0)
