#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:46:26 2022

@author: hauke
"""

from .baseconfig import BaseConfig
from custom_class import Landscape

class PerlinConfig(BaseConfig):
    landscape = Landscape("Perlin_uniform", stdE=5., stdI=5., connection_probability=.2, shift=1., params={"size": 4}, seed=0)
