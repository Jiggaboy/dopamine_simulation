#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:23:24 2022

@author: hauke
"""


from dataclasses import dataclass, field

from lib.connectivity_landscape import SYMMETRIC_LANDSCAPES, independent


@dataclass(frozen=True)
class Landscape:
    """
    mode: random, independent, Perlin, Perlin_uniform, symmetric
    stdE/stdI: the width of the connectivity distribution in the respective grid
    connection_probability: determines the out-degree of a neuron
    shift: the shift of connections in the preferred direction
    params: parameter about the mode
    seed: of the random generator

    properties:
        is_asymmetric
        is_independent
    """
    mode: str
    stdE: float
    stdI: float
    connection_probability: float
    shift: float = 0.
    params: dict = field(default_factory=dict)
    seed: int = None

    @property
    def is_asymmetric(self)->bool:
        return self.mode not in SYMMETRIC_LANDSCAPES

    @property
    def is_independent(self)->bool:
        return self.mode == independent
