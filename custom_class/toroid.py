#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:00:06 2021

@author: hauke
"""

import numpy as np


from collections import namedtuple
Coordinate = namedtuple("Coordinate", ("x1", "x2"))


class Toroid():

    def __init__(self, shape:tuple, def_value=0):
        self.space = np.full(shape, fill_value=def_value, dtype=int)
        self.height = shape[1]
        self.width = shape[0]


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



### TEST
import unittest

HEIGHT = 100
WIDTH = 125



class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.torus = Toroid((HEIGHT, WIDTH))
        cls.torus = Toroid((WIDTH, HEIGHT))

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass
    def tearDown(self):
        pass


    def test_grid(self):
        self.assertEqual(self.torus.height, HEIGHT)
        self.assertEqual(self.torus.width, WIDTH)


    def test_distance_values(self):
        DELTA = 0.001

        p1 = Coordinate(1, 1)
        p2 = Coordinate(2, 3)
        p3 = Coordinate(self.torus.height - 1, 1)
        p4 = Coordinate(3, self.torus.width - 1)
        self.assertAlmostEqual(self.torus.get_distance(p1, p2), np.sqrt(5), delta=DELTA)
        self.assertAlmostEqual(self.torus.get_distance(p3, p2), np.sqrt(13), delta=DELTA)
        self.assertAlmostEqual(self.torus.get_distance(p2, p3), np.sqrt(13), delta=DELTA)
        self.assertAlmostEqual(self.torus.get_distance(p1, p4), np.sqrt(8), delta=DELTA)

        self.assertAlmostEqual(self.torus.get_distance(p2, p3, form="squared"), 13, delta=DELTA)
        self.assertAlmostEqual(self.torus.get_distance(p1, p4, form="squared"), 8, delta=DELTA)

        pn = (-1, -1)
        self.assertAlmostEqual(self.torus.get_distance(p2, pn), np.sqrt(25))
        self.assertAlmostEqual(self.torus.get_distance(p3, pn), np.sqrt(4))

        po = (HEIGHT, WIDTH)
        self.assertAlmostEqual(self.torus.get_distance(p2, po), np.sqrt(13))
        self.assertAlmostEqual(self.torus.get_distance(p3, po), np.sqrt(2))



    def test_distance_types(self):
        p2 = Coordinate(2, 3)
        p3 = np.array([3, 4])
        p5 = (2, 3)
        self.assertEqual(self.torus.get_distance(p2, p5), 0)
        self.assertAlmostEqual(self.torus.get_distance(p2, p3), np.sqrt(2))
        self.assertEqual(self.torus.get_distance(p2, p5), 0)


    def test_distance_raises(self):
        with self.assertRaises(ValueError):
            p1 = Coordinate(1, 1)
            p2 = Coordinate(2, 3)
            self.torus.get_distance(p1, p2, form="error")



if __name__ == '__main__':
    import datetime
    print(datetime.datetime.now())
    unittest.main()
