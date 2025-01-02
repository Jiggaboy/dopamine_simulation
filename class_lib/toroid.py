#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: Functionality of a Torus.

Requires the Data directory from the baseconfig/constants.

"""

import numpy as np
from pathlib import Path

from constants import DATA_DIR

distances_filename = "distances_{width}_{height}"

class Toroid():

    def __init__(self, shape:(int, tuple), def_value=-1):
        if isinstance(shape, int):
            shape = (shape, shape)
        self.space = np.full(shape, fill_value=def_value, dtype=int)
        self.height = shape[1]
        self.width = shape[0]


    @property
    def filename(self) -> str:
        return Path(DATA_DIR).joinpath(distances_filename.format(width=self.width, height=self.height))


    # __getitem__/__setitem__ allows for .-notation
    def __getitem__(self, coor:tuple):
        return self.space[coor]


    def __setitem__(self, coor:tuple, value:int):
        self.space[coor] = value


    def get_distance(self, p1:tuple, p2:tuple=None, form:str="norm")->float:
        """form: 'norm' or 'squared'"""
        p1 = np.asarray(p1)
        if p2 is None:
            p2 = (0, 0)
        p2 = np.asarray(p2)
        height_difference, width_difference = np.abs(p1 - p2)

        vector = np.zeros(2, dtype=int)
        vector[0] = np.min((height_difference, (self.height - height_difference)))
        vector[1] = np.min((width_difference, (self.width - width_difference)))

        form = form.lower()
        if form == 'squared':
            distance = vector @ vector.T
        elif form == 'norm':
            distance = np.linalg.norm(vector)
        else:
            raise ValueError("Argument form not properly defined. Valid: 'norm', 'squared'.")
        return distance


    def get_distances(self, force_calculation:bool=False):
        """Also saves the distances in a file."""
        if not force_calculation:
            try:
                return np.loadtxt(self.filename)
            except FileNotFoundError:
                pass

        distances = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                distances[i, j] = self.get_distance((i, j))

        np.savetxt(self.filename, distances)
        return distances
