#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""


from dataclasses import dataclass

@dataclass
class SequenceCounter:
    tag: str
    center: tuple


    def __len__(self):
        return len(self.center)
