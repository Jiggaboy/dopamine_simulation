#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:07:51 2022

@author: hauke
"""


from dataclasses import dataclass


@dataclass
class ExternalDrive:
    mean: float
    std: float
    seeds: tuple
