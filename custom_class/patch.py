#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:31:58 2022

@author: hauke
"""

from dataclasses import dataclass

@dataclass
class Patch:
    radius: float
    amount_of_neurons: int
    weicht_change: float
    fraction_synapses: float
