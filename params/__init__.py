#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""
from .baseconfig import BaseConfig
from .motifconfig import *

# Need to be after the Params imports, otherwise circular import possible
from .config_handler import config
