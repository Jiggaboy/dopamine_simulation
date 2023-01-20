#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    PerlinConfig
    Data of the corresponding config with the patches starter and repeater

"""


import unittest as UT
import numpy as np
import matplotlib.pyplot as plt

from analysis.lib import SubspaceAngle

from params import PerlinConfig
import dopamine as DP



class TestSubspaceAngle(UT.TestCase):
    cfg = PerlinConfig()

    tag = "repeater", "starter"
    MAX_components = 5
    THR_components = 1000

    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(29, 29), radius=24)

    @property
    def seed(self):
        return self.cfg.drive.seeds[0]


    def setUp(self):
        self.tags = self.cfg.get_all_tags(self.tag, seeds=self.seed)
        self.angle = SubspaceAngle(self.cfg)

    @UT.skip("Work in progress...")
    def test_patch_vs_baseline(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(self.tags[0], self.cfg.baseline_tag(self.seed), n_components=i)



    @UT.skip("Work in progress...")
    def test_patch_vs_patch(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(*self.tags[:2], n_components=i)



    @UT.skip("Work in progress...")
    def test_local(self):
        for i in range(1, self.MAX_components):
            self.angle.fit(self.tags[0], self.cfg.baseline_tag(self.seed), n_components=i, mask=self.LOCAL_NEURONS)


    def test_alignment_index(self):

        import sklearn.decomposition as sk
        DIM = 2
        N_SAMPLES = 1000

        def create_nD_noise(n:int=2):
            return np.random.normal(size=(N_SAMPLES, n))

        def create_noise():
            return np.random.normal(size=(N_SAMPLES, DIM-1))


        def create_data(x1:float=1, x2:float=1, y1:float=1, y2:float=1):
            data1 = create_nD_noise()
            data1[:, 0]  *= x1
            data1[:, 1]  *= x2
            data2 = create_nD_noise()
            data2[:, 0]  *= y1
            data2[:, 1]  *= y2
            return data1, data2


        def create_3D_data(x1:float=1, x2:float=1, x3:float=1, y1:float=1, y2:float=1, y3:float=1):
            data1 = create_nD_noise(n=3)
            data2 = create_nD_noise(n=3)
            for i, (x, y) in enumerate(zip([x1, x2, x3], [y1, y2, y3])):
                data1[:, i]  *= x
                data2[:, i]  *= y
            return data1, data2

        def plot_2D(data1, data2):
            fig, ax = plt.subplots()
            ax.scatter(*data1.T)
            ax.scatter(*data2.T)
            lim = data1.max()
            LIM = (-lim, lim)
            ax.set_xlim(LIM)
            ax.set_ylim(LIM)

        def plot_3D(data1, data2):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(*data1.T)
            ax.scatter(*data2.T)
            lim = data1.max()
            LIM = (-lim, lim)
            ax.set_xlim(LIM)
            ax.set_ylim(LIM)
            ax.set_zlim(LIM)


        def get_alignment_index(data1, data2):
            pca1 = sk.PCA()
            pca1.fit(data1)
            pca2 = sk.PCA()
            pca2.fit(data2)
            self.angle.min_components_ = data1.shape[1]
            ai = self.angle._full_alignment_indexes(pca1, pca2)
            ai_rev = self.angle._full_alignment_indexes(pca2, pca1)
            return ai, ai_rev


        def rotate_2D(data, phi:float):
            theta = np.radians(phi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            print(f"Rotate by {phi} degree")
            return R.dot(data.T).T


        def analyze_data(data1, data2, scales):
            if data1.shape[1] == 2:
                plot_2D(data1, data2)
            else:
                plot_3D(data1, data2)

            print(f"Scales are:{scales}")
            print(get_alignment_index(data1, data2))


        scales = [
            (1, 1, 1, 1),
            (10, 1, 10, 1),
            (10, 1, 5, 1),
            (10, 1, 1, 10),
            (10, 3, 5, 10),
            (3, 10, 5, 10),
            (12, 3, 5, 10),
        ]

        for s in scales:
            data1, data2 = create_data(*s)
            analyze_data(data1, data2, s)
            data1 = rotate_2D(data1, 30)
            analyze_data(data1, data2, s)




        scales_3D = [
            (1, 1, 1, 1, 1, 1),
            (10, 2, 1, 5, 4, 3),
            (4, 6, 2, 5, 4, 3),
            (1, 2, 3, 5, 4, 3),
        ]

        for s in scales_3D:
            data1, data2 = create_3D_data(*s)
            analyze_data(data1, data2, s)
            data1[:, :2] = rotate_2D(data1[:, :2], 30)
            analyze_data(data1, data2, s)


        plt.show()







if __name__ == '__main__':
    UT.main()
