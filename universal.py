#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:59:39 2021

@author: hauke
"""

import numpy as np


def set_seed(use_constant_seed:bool=None):
    if use_constant_seed:
        np.random.seed(0)
    else:
        np.random.seed(None)


def get_filename(*tags):
    return "_".join((str(t) for t in tags))
