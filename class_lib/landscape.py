#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

from dataclasses import dataclass, field
from collections import OrderedDict
from lib.connectivity_landscape import independent


@dataclass
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
    stdE: float = None
    stdI: float = None
    connection_probability: float = .1
    shift: float = 0.
    params: dict = field(default_factory=dict)
    seed: int = None


    def __post_init__(self):
        if not self.is_independent:
            assert self.stdE is not None
            assert self.stdI is not None

    @property
    def is_independent(self)->bool:
        return self.mode == independent


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
