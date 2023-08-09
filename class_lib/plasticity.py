#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:33:09 2022

@author: hauke
"""


from dataclasses import dataclass


@dataclass(frozen=True)
class Plasticity:
    rate: float
    cap: float = 2.
