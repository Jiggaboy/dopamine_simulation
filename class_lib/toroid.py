#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hauke Wernecke
"""

import numpy as np

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
        return distances_filename.format(width=self.width, height=self.height)



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




### TEST
import unittest

HEIGHT = 100
WIDTH = 125



class TestModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        p1 = (1, 1)
        p2 = (2, 3)
        p3 = (self.torus.height - 1, 1)
        p4 = (3, self.torus.width - 1)
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
        p2 = (2, 3)
        p3 = np.array([3, 4])
        p5 = (2, 3)
        self.assertEqual(self.torus.get_distance(p2, p5), 0)
        self.assertAlmostEqual(self.torus.get_distance(p2, p3), np.sqrt(2))
        self.assertEqual(self.torus.get_distance(p2, p5), 0)


    def test_distance_raises(self):
        with self.assertRaises(ValueError):
            p1 = (1, 1)
            p2 = (2, 3)
            self.torus.get_distance(p1, p2, form="error")


    def test_get_distances(self):
        side = 20
        torus = Toroid((side, side))
        distances = torus.get_distances()

        self.assertEqual(distances.shape, (side, side))
        import matplotlib.pyplot as plt
        plt.figure("unshifted")
        plt.imshow(distances)
        center = (3, 5)
        plt.figure("shifted")
        shifted = np.roll(distances, center, axis=(0, 1))
        plt.imshow(shifted)

        plt.figure("patch")
        plt.imshow(shifted < 5)

if __name__ == '__main__':
    import datetime
    print(datetime.datetime.now())
    unittest.main()
