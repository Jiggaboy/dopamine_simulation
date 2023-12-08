#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Test requirements:
    dbscan


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'


import unittest as UT

from analysis.lib.dbscan import DBScan

import numpy as np
import matplotlib.pyplot as plt


from plot.sequences import _plot_cluster
EPS = 4.5
SAMPLES = 3

spikes_per_timepoint = 30
nrows = 100
t = 200

PLOT = True

##### Test cluster algorithm with another method: Calculating the distance with the time and torus seperatily
# from class_lib import Toroid

# torus = Toroid(shape=self._config.rows)
# ###########################################
# def _get_distance(X0, X1):
#     time_distance = np.abs(X0[0] - X1[0])
#     torus_distance = torus.get_distance(X0[1:], X1[1:])
#     return time_distance + torus_distance

class TestDBScan(UT.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.dbscan = DBScan(eps=EPS, min_samples=SAMPLES)


    def tearDown(self):
        plt.show()


    def test_sample_spike_train(self):
        nrows = 80
        filename = "test/sample_spike_train.npy"
        spike_train = np.load(filename)

        time_low = 1780
        time_high = 1820
        time_mask = np.logical_and(spike_train[:, 0] < time_high, spike_train[:, 0] > time_low)
        spikes = spike_train[time_mask]

        ### Check with explicit distance calculation method (see above).
        # db_explicit_method = DBScan(eps=eps, min_samples=min_samples, n_jobs=-1, metric=_get_distance)
        # data_explicit_method, labels_explicit_method = db_explicit_method.fit(spike_train, nrows=80)

        # Run the clustering with all data points before any filtering of labels, or spaces.
        db = DBScan(eps=EPS, min_samples=SAMPLES, n_jobs=-1)
        data, labels = db.fit_toroidal(spikes, nrows=nrows) # Original data on a 80x80 grid

        space_mask = np.logical_and.reduce((data[:, 1] > 17, data[:, 1] < 21,
                                            data[:, 2] > 58, data[:, 2] < 62))
        ## Find all those clusters which cross the spot at some point.
        seq_ids = set(labels[space_mask])
        print(seq_ids)
        ### Filter the data and labels for the specific sequences.
        # seq_mask = np.isin(labels, list(seq_ids))
        # data = data[seq_mask]
        # labels = labels[seq_mask]

        # Filter the space again
        filter_mask = np.logical_and.reduce((data[:, 1] > 52, data[:, 1] < 55,
                                            data[:, 2] > 0, data[:, 2] < 10))

        data_ = data[filter_mask]
        labels_ = labels[filter_mask]

        ## Check whether a new scan reveals the same cluster?
        db_ = DBScan(eps=EPS, min_samples=SAMPLES, n_jobs=-1)
        data_, labels_ = db_.fit_toroidal(data_, nrows=nrows)
        _plot_cluster(data_, labels_)

        # Check whether they are purely linked in time
        # Get all the coordinates
        spike_coordinates = set(tuple(d) for d in data_[:, 1:])
        for c in spike_coordinates:
            idx = (data_[:, 1:] == c).all(axis=1)
            time_diff = np.diff(data_[:, 0][idx])
            print(c, time_diff[time_diff > 1])

        try:
            for s in seq_ids:
                data_seq = data[labels == s]
                labels_seq = labels[labels == s]
                _plot_cluster(data_seq, labels_seq)
        except NameError:
            pass

        _plot_cluster(data, labels)
        plt.show()



    @UT.skip("Addings test of link labels")
    def test_merge_labels(self):
        """
        Several test cases for the merge_labels-method:
            After the spikes are determined, they got clustered and receive labels.
            Thus each label correspond to a spike in time.

        """
        ##### Noisy spikes
        labels = np.asarray([-1, -1])
        labels_shifted = np.asarray([-1, -1])
        merged_labels = self.dbscan.merge_labels(labels, labels_shifted)
        expectation = np.asarray([-1, -1])
        self.assertTrue((expectation == merged_labels).all(), merged_labels)


        ##### A big cluster that crosses borders in the shifted and unshifted case
        labels = np.asarray([0, 0, 0, 1, 1])
        labels_shifted = np.asarray([0, 0, 1, 1, 1])
        merged_labels = self.dbscan.merge_labels(labels, labels_shifted)
        expectation = np.asarray([0, 0, 0, 0, 0])
        self.assertTrue((expectation == merged_labels).all(), merged_labels)
        # Reversed
        merged_labels = self.dbscan.merge_labels(labels_shifted, labels)
        self.assertTrue((expectation == merged_labels).all(), merged_labels)

        ##### A Cluster that separates into 4 small clusters when shifted (and vice versa)
        labels = np.asarray([0, 0, 0 ,0])
        labels_shifted = np.asarray([0, 1, 2, 3])
        merged_labels = self.dbscan.merge_labels(labels, labels_shifted)
        expectation = np.asarray([0, 0, 0, 0])
        self.assertTrue((expectation == merged_labels).all())
        # Reversed
        merged_labels = self.dbscan.merge_labels(labels_shifted, labels)
        self.assertTrue((expectation == merged_labels).all())
        # Shuffled
        shuffled = labels_shifted.copy()
        np.random.shuffle(shuffled)
        merged_labels = self.dbscan.merge_labels(shuffled, labels)
        self.assertTrue((expectation == merged_labels).all(), merged_labels)


        ##### A Cluster that separates into 4 small noisy clusters when shifted (and vice versa)
        labels = np.asarray([0, 0, 0 ,0, 1, 1, 2, 3])
        labels_shifted = np.asarray([-1, -1, -1, -1, -1, -1, 0, 1])
        merged_labels = self.dbscan.merge_labels(labels, labels_shifted)
        expectation = np.asarray([0, 0, 0, 0])
        self.assertTrue((expectation == merged_labels).all(), merged_labels)
        # Reversed
        merged_labels = self.dbscan.merge_labels(labels_shifted, labels)
        self.assertTrue((expectation == merged_labels).all(), merged_labels)


        ##### A Cluster that separates into 1big and one noisy cluster
        labels = np.asarray([0, 0, 0 ,0, 0, 0, 0, 0])
        labels_shifted = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        merged_labels = self.dbscan.merge_labels(labels, labels_shifted)
        expectation = np.asarray([0, 0, 0 ,0, 0, 0, 0, 0])
        self.assertTrue((expectation == merged_labels).all(), merged_labels)
        # Reversed
        merged_labels = self.dbscan.merge_labels(labels_shifted, labels)
        self.assertTrue((expectation == merged_labels).all(), merged_labels)
        # Shuffled
        shuffled = labels_shifted.copy()
        np.random.shuffle(shuffled)
        merged_labels = self.dbscan.merge_labels(shuffled, labels)
        self.assertTrue((expectation == merged_labels).all(), merged_labels)
        pass





    @UT.skip("Addings test of link labels")
    def test_static_cluster(self):
        clusters = self._create_cluster(static=True)
        spikes = np.hstack(clusters)
        # Create static timeline and stack them on the spikes
        times = np.repeat(np.arange(t), spikes.shape[-1])
        t_spikes = np.tile(spikes, t)
        data = np.vstack([times, t_spikes]).T

        data, labels = self.dbscan.fit_toroidal(data, nrows=nrows, remove_noisy_data=False)
        if PLOT:
            self._plot_cluster(data, labels)
            plt.title("Frozen noise")


    @UT.skip("Addings test of link labels")
    def test_nonstatic_cluster(self):
        clusters = self._create_cluster(static=False)
        data = self._stack_spikes_times(t, clusters)

        data_, labels = self.dbscan.fit_toroidal(data, nrows=nrows, remove_noisy_data=False)
        if PLOT:
            self._plot_cluster(data_, labels)
            plt.title("Non-static noise")


    @UT.skip("Addings test of link labels")
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


    @UT.skip("Addings test of link labels")
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


    @UT.skip("Addings test of link labels")
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
    plt.show()
