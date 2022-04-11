#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:51:48 2022

@author: hauke
"""


from dataclasses import dataclass


@dataclass(frozen=True)
class Network:
    rows: int
    synaptic_weight: float
    ei_factor: float

    # External drive
    drive_mean: float
    drive_std: float

    # Transfer function
    tf_steepness: float
    tf_offset: float


    @property
    def exc_weight(self):
        return self.synaptic_weight


    @property
    def inh_weight(self):
        return - self.ei_factor * self.synaptic_weight
