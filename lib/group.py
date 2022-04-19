#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:08:38 2022

@author: hauke
"""


from dataclasses import dataclass

@dataclass
class Group:
    rows: int
    std: float

    def __post_init__(self):
        self.quantity = self.rows ** 2
