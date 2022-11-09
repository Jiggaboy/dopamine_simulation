#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-26

@author: Hauke Wernecke

Test requirements:
    create_image
    some data

"""


import unittest as UT

from animation.activity import create_image, activity

import numpy as np
import matplotlib.pyplot as plt

from lib import pickler as PIC


SQUARE_LEN = 5

PLOT = True

class TestPlot(UT.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        plt.show()


    @UT.skip("Test activity right now.")
    def test_create_image(self):
        """
        The image should appear in both plots.
        """
        data = self.create_data()

        fig, axes = plt.subplots(nrows=2)
        # Use the current active axis (lower plot)
        create_image(data, norm=(0, data.size))
        # Specify axis
        create_image(data, norm=(0, data.size), axis=axes[0])


    def test_activity(self):
        plain_data = self.create_data()
        activity(plain_data, norm=(0, plain_data.size), title="Single activation")


        shuffled = self.create_data()
        np.random.shuffle(shuffled)
        data = [plain_data, shuffled]
        activity(*data, norm=(0, data[0].size))



    @staticmethod
    def create_data():
        return np.arange(SQUARE_LEN**2)




if __name__ == '__main__':
    UT.main()
