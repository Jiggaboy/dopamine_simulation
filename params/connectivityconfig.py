#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:07:50 2022

@author: hauke
"""



import numpy as np
from collections import OrderedDict
from .baseconfig import BaseConfig

from custom_class import Landscape


class ConnectivityConfig(BaseConfig):
    rows = 30

    landscape = Landscape("Perlin_uniform", stdE=3., stdI=3., connection_probability=5., shift=1., params={"size": 3, "base": 3}, seed=0)
    # landscape = Landscape("symmetric", params={"size": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("homogeneous", params={"phi": 2, "stdE": 3., "stdI": 2.})
    # landscape = Landscape("random", params={"stdE": 3., "stdI": 2.})
