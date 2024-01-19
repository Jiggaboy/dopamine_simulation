#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""


from dataclasses import dataclass


@dataclass
class Synapse:
    weight: float
    EI_factor: float

    @property
    def J(self)->float:
        return self.weight

    @property
    def g(self)->float:
        return self.EI_factor

    @property
    def exc_weight(self)->float:
        return self.weight

    @property
    def inh_weight(self)->float:
        return -1 * self.EI_factor * self.weight

    @property
    def values(self)->tuple:
        return self.J, self.g
