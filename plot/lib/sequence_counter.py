#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:32:12 2022

@author: hauke
"""


from dataclasses import dataclass

@dataclass
class SequenceCounter:
    tag: str
    center: tuple
