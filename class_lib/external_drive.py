#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""


from dataclasses import dataclass


@dataclass
class ExternalDrive:
    mean: float
    std: float
    seeds: tuple
