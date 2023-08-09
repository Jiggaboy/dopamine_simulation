#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-30

@author: Hauke Wernecke
"""


from dataclasses import dataclass, field

@dataclass
class AngleDumper:
    tag: str
    center: tuple
    radius: tuple
    n_components: int
    angles: dict = field(default_factory=dict)