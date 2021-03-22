#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:57:29 2021

@author: hauke
"""

from dataclasses import dataclass

@dataclass
class Circle():
    pos: tuple
    radius: float
