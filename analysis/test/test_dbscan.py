#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

Test requirements:
    dbscan

"""


import unittest as UT

from analysis.lib.dbscan import DBScan

import numpy as np
import matplotlib.pyplot as plt


EPS = 4
SAMPLES = 20

spikes_per_timepoint = 30
nrows = 100
t = 200

PLOT = True

class TestDBScan(UT.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.dbscan = DBScan(eps=EPS, min_samples=SAMPLES)


    def tearDown(self):
        plt.show()



    def test_static_cluster(self):
        clusters = self._create_cluster(static=True)
        spikes = np.hstack(clusters)
        # Create static timeline and stack them on the spikes
        times = np.repeat(np.arange(t), spikes.shape[-1])
        t_spikes = np.tile(spikes, t)
        data = np.vstack([times, t_spikes]).T

        data_, labels = self.dbscan.fit_toroidal(data, nrows=nrows, remove_noisy_data=False)
        if PLOT:
            self._plot_cluster(data_, labels)
            plt.title("Frozen noise")


    def test_nonstatic_cluster(self):
        clusters = self._create_cluster(static=False)
        data = self._stack_spikes_times(t, clusters)

        data_, labels = self.dbscan.fit_toroidal(data, nrows=nrows, remove_noisy_data=False)
        if PLOT:
            self._plot_cluster(data_, labels)
            plt.title("Non-static noise")


    def test_modulated_cluster(self):
        clusters = self._create_cluster(static=False)

        # Add some modulation in x and/or y direction
        clusters[0, 0] += self._modulation(t, amplitude=8)
        clusters[0, 1] += self._modulation(t, amplitude=-6)

        clusters[2, 1] += self._modulation(t, amplitude=-4)

        clusters[5, 0] += self._modulation(t, amplitude=4)
        clusters[5, 1] += self._modulation(t, amplitude=12)

        clusters[6, 0] += self._modulation(t, amplitude=6)
        clusters[6, 1] += self._modulation(t, amplitude=8)
        data = self._stack_spikes_times(t, clusters)

        data_, labels = self.dbscan.fit_toroidal(data, nrows=nrows, remove_noisy_data=False)
        if PLOT:
            self._plot_cluster(data_, labels)
            plt.title("Sinus-modulated noise")


    def test_time_sep_clusters(self):
        X = Y = 25
        SIGMA = 4
        duration = 10
        cluster1 = self._sample_cluster(x=X, y=Y, sigma=SIGMA, number=spikes_per_timepoint, t=duration)

        data1 = self._stack_spikes_times(duration, np.array([cluster1, ]))
        data2 = self._stack_spikes_times(duration, np.array([cluster1, ]))
        data2[:, 0] += nrows // 2
        data3 = self._stack_spikes_times(duration, np.array([cluster1, ]))
        data3[:, 0] += nrows

        joint_data = np.vstack([data1, data2, data3])

        data_, labels = self.dbscan.fit_toroidal(joint_data, nrows=nrows)
        if PLOT:
            self._plot_cluster(data_, labels)
            plt.title("Separated clusters")


    def _create_cluster(self, static:bool)->np.ndarray:
        """
        Creates clusters which may vary over time (static-parameter).
        Some are in the center and others at the edge of the grid.
        """
        time = None if static else t
        cluster_bottom_left = self._sample_cluster(x=30, y=30, sigma=4, number=spikes_per_timepoint, t=time)
        cluster_center = self._sample_cluster(x=50, y=55, sigma=5, number=spikes_per_timepoint, t=time)
        cluster_top_right = self._sample_cluster(x=80, y=75, sigma=4, number=spikes_per_timepoint, t=time)
        cluster_bottom_right = self._sample_cluster(x=80, y=25, sigma=7, number=spikes_per_timepoint, t=time)
        cluster_top_left = self._sample_cluster(x=25, y=75, sigma=3, number=spikes_per_timepoint, t=time)

        cluster_corner = self._sample_cluster(x=0, y=5, sigma=3, number=spikes_per_timepoint, t=time)
        cluster_edge_hor = self._sample_cluster(x=50, y=1, sigma=4, number=spikes_per_timepoint, t=time)
        cluster_edge_ver = self._sample_cluster(x=3, y=60, sigma=4, number=spikes_per_timepoint, t=time)

        return np.asarray([
            cluster_bottom_left,
            cluster_center,
            cluster_top_right,
            cluster_bottom_right,
            cluster_top_left,
            cluster_corner,
            cluster_edge_hor,
            cluster_edge_ver,
        ])


    def _sample_cluster(self, x:float, y:float, sigma:float, number:int, t:int=None):
        """
        Create clouds with the same sigma for different x and y coordinates.
        """
        size = number if not t else (number, t)
        return np.array([np.random.normal(loc=p, scale=sigma, size=size) for p in (x, y)]) % nrows


    @staticmethod
    def _plot_cluster(data:np.ndarray, labels:np.ndarray=None, force_label:int=None):
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection="3d")
        ax.set_xlabel("time")
        ax.set_ylabel("X-Position")
        ax.set_zlabel("Y-Position")

        if labels is None:
            ax.scatter(*data.T, marker=".")
            return

        unique_labels = np.unique(labels)
        print(unique_labels)
        for l in unique_labels:
            if force_label is not None and l != force_label:
                continue
            ax.scatter(*data[labels == l].T, label=l, marker=".")
        plt.legend()


    @staticmethod
    def _stack_spikes_times(max_time:int, clusters:np.ndarray):
        spikes = np.hstack(clusters)
        spikes = np.moveaxis(spikes, 1, -1)
        times = np.repeat(np.arange(max_time), spikes.shape[-1])
        spikes = spikes.reshape(spikes.shape[0], (spikes.shape[1] * spikes.shape[2]))
        return np.vstack([times, spikes]).T


    @staticmethod
    def _modulation(time:np.ndarray, amplitude:float):
        return amplitude * np.sin(np.arange(time) / 2 / np.pi)





if __name__ == '__main__':
    UT.main()
