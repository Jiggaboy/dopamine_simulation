#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:43:07 2021

@author: hauke
"""

import numpy as np
import noise


# def move(nrow):
#     return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])


def shift(direction:np.ndarray, bins=8)->np.ndarray:
    nrows = direction.size
    s = np.zeros((nrows, 2), dtype=int)
    cos = np.cos(direction / bins * 2 * np.pi)
    s[cos > 0.1, 0] = 1
    s[cos < -0.1, 0] = -1

    sin = np.sin(direction / bins * 2 * np.pi)
    s[sin > 0.1, 1] = 1
    s[sin < -0.1, 1] = -1

    return s



def homogeneous(nrow, phi=4, **kwargs):
    npop = np.power(nrow, 2)
    landscape = np.full(npop, fill_value=phi, dtype=int)
    return landscape


def random(nrow, seed=0, **kwargs):
    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.randint(8, size=npop)
    return landscape


def Perlin(nrow, size=5, base=0, **kwargs):
    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size, base=base)
         for j in y] for i in x]
    m = n - np.min(n)
    m /= m.max()
    return m.ravel()


def Perlin_uniform(nrow, *args, **kwargs):
    """Creates a Perlin configuration and split them into 8 uniform bins between [0, 1]."""
    BINS = 8
    m = Perlin(nrow, *args, **kwargs)
    a = np.argsort(m)
    b = np.power(nrow, 2) // BINS
    for j, i in enumerate(np.linspace(0, 1, BINS, endpoint=False)):
        m[a[j * b:(j + 1) * b]] = i
    return m



def calc_plot_shift(callable_:callable=None, X=None, Y=None, name:str=None, *args, **kwargs):
    plt.figure(name)
    if callable_ is not None:
        D = callable_(*args, **kwargs)
    U, V = calculate_direction(D, **kwargs)
    plt.quiver(X, Y, U, V, pivot='middle')


def plot_shift(X=None, Y=None, D=None, name:str=None, **kwargs):
    plt.figure(name)
    U, V = D[:, 0] / np.linalg.norm(D, axis=1), D[:, 1] / np.linalg.norm(D, axis=1)
    plt.quiver(X, Y, U, V, pivot='middle')


## TESTING

import unittest
import matplotlib.pyplot as plt


def calculate_direction(x, bins=8, **kwargs):
    rad = 2 * np.pi
    u = np.cos(x / bins * rad)
    v = np.sin(x / bins * rad)
    return u, v


class TestConfiguration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = y = np.arange(35)
        cls.nrows = x.size

        X, Y = np.meshgrid(x, y)
        cls.X, cls.Y = X.ravel(), Y.ravel()

    @classmethod
    def tearDownClass(cls):
        pass


    def show_perlin(self, callable_:callable, *args, **kwargs):
        plt.figure()
        data = callable_(self.nrows, *args, **kwargs)
        data = data.reshape((self.nrows, self.nrows))
        plt.imshow(data)
        plt.colorbar()


    def hist_distribution(self, callable_:callable, *args, **kwargs):
        plt.figure()
        data = callable_(self.nrows, *args, **kwargs)
        plt.hist(data, bins=8)


    def test_homogenous(self):
        calc_plot_shift(homogeneous, self.X, self.Y, "Homo", self.nrows, phi=1)
        self.hist_distribution(homogeneous, phi=3)


    def test_random(self):
        calc_plot_shift(random, self.X, self.Y, "Random", self.nrows)
        self.hist_distribution(random)


    def test_perlin(self):
        size_ = 10
        calc_plot_shift(Perlin, self.X, self.Y, "Perlin", self.nrows, size=size_, bins=1)
        self.hist_distribution(Perlin, size=size_)
        self.show_perlin(Perlin, size=size_)


    def test_perlin_uniform(self):
        size_ = 2
        calc_plot_shift(Perlin_uniform, self.X, self.Y, "Perlin Uni", self.nrows, size=size_, bins=1)
        self.hist_distribution(Perlin_uniform, size=size_)
        self.show_perlin(Perlin_uniform, size=size_)


    # def test_shift(self):
    #     print(shift(np.arange(8)))


if __name__ == '__main__':
    unittest.main()
