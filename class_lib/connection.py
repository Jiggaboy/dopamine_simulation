#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""


from dataclasses import dataclass

from . import Group

@dataclass
class Connection:
    idx: int
    source: Group
    target: Group
    connection_probability: float
    allow_self_connections: bool = None

    @property
    def no_of_targets(self)->int:
        return int(self.connection_probability * self.target.quantity)

    @property
    def std(self)->float:
        return self.source.std


    def get_all(self):
        return self.idx, self.source.rows, self.target.rows, self.no_of_targets, self.std, self.allow_self_connections
