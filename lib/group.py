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

    @property
    def quantity(self):
        return self.rows ** 2
