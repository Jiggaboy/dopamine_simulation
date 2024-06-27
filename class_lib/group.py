#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A dataclass that defines a group of neuron that may be connected to one another.

@author: Hauke Wernecke
"""


from dataclasses import dataclass

@dataclass
class Group:
    rows: int
    std: float

    @property
    def quantity(self):
        return self.rows ** 2
