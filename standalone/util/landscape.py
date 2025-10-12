#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

from dataclasses import dataclass, field
from collections import OrderedDict
from util.connectivity_landscape import SYMMETRIC_LANDSCAPES


@dataclass
class Landscape:
    """
    mode: random, Perlin, Perlin_uniform, symmetric
    stdE/stdI: the width of the connectivity distribution in the respective grid
    connection_probability: determines the out-degree of a neuron
    shift: the shift of connections in the preferred direction
    params: parameter about the mode
    seed: of the random generator

    properties:
        is_asymmetric
    """
    mode: str
    stdE: float = None
    stdI: float = None
    connection_probability: float = .1
    shift: float = 0.
    params: dict = field(default_factory=dict)
    seed: int = None

    @property
    def is_asymmetric(self)->bool:
        return self.mode not in SYMMETRIC_LANDSCAPES


    @property
    def info(self)->dict:
        params = OrderedDict()
        params["mode"] = self.mode
        params["stdE"] = self.stdE
        params["stdI"] = self.stdI
        params["shift"] = self.shift
        params["connection_probability"] = self.connection_probability
        params["seed"] = self.seed
        return params
