#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:43:41 2022

@author: Hauke Wernecke
"""
from .baseconfig import BaseConfig
from .brianconfig import *
from .motifconfig import *
from .perlinconfig import PerlinConfig
from .testconfig import TestConfig

# Need to be after the Params imports, otherwise circular import possible
from .config_handler import config
