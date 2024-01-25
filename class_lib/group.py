#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
