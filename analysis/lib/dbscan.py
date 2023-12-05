#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-11

@author: Hauke Wernecke

Questions:
    Why not calculating the distance matrix (being smart about the time succession)?
"""

import cflogger
logger = cflogger.getLogger()

import numpy as np
import sklearn.cluster as cluster

from lib import functimer


class DBScan(cluster.DBSCAN):
    """
    Inherits from sklearn.cluster.DBSCAN.
    """
    NOISE_LABEL = -1


    @functimer(logger=logger)
    def fit(self, data:np.ndarray, remove_noisy_data:bool=True):
        """
        Performs a dbscan and returns the labels.

        Parameter:
            data: np.ndarray of shape (n, 3) with n data points. Each point is aggregated by (time, x, y)
        """
        db = super().fit(data)
        labels = db.labels_
        if remove_noisy_data:
            data, labels = self._remove_noise_labels(data, labels)
        return data, labels


    @functimer(logger=logger)
    def fit_toroidal(self, data:np.ndarray, nrows:int, remove_noisy_data:bool=True):
        """

        Note: labels refers to the label each data point has according to the cluster id
        Note: cluster refers to all data points with the same label (or a pairwise combination of labels)
        """
        data_shifted = self._recenter_data(data, nrows)

        # Perform dbscan on original and shifted space
        _, labels = self.fit(data, remove_noisy_data=False)
        _, labels_shifted = self.fit(data_shifted, remove_noisy_data=False)


        ## test - start
        data_ = data.copy()
        # mask = np.logical_or(labels == 10, labels == 14)
        # data_ = data_[mask]
        # labels = labels[mask]
        # labels_shifted = labels_shifted[mask]
        labels_ = self.merge_labels(labels, labels_shifted)
        if remove_noisy_data:
            logger.info("Remove noise labels of the data.")
            data_, labels_ = self._remove_noise_labels(data, labels_)
        self.data = data_
        self.labels = labels_
        return data_, labels_
        ## test - end

        # link the labels of the clusters pairwise
        pairwise_linked_labels = self._link_labels(labels, labels_shifted)
        unique_labels = self._omit_joint_clusters_in_unshifted_space(pairwise_linked_labels)

        for u_label in reversed(unique_labels):
            joint_labels = self._get_splitted_clusterlabels(pairwise_linked_labels, u_label)
            self._relabel_splitted_clusters(labels, labels_shifted, joint_labels, u_label)


        if remove_noisy_data:
            logger.info("Remove noise labels of the data.")
            data, labels = self._remove_noise_labels(data, labels)

        logger.debug(f"# clusters: {len(np.unique(labels))}")
        self.data = data
        self.labels = labels
        return data, labels


    def merge_labels(self, labels, labels_shifted):
        labels_tmp = labels.copy()
        labels_shifted_tmp = labels_shifted.copy()

        # link the labels of the clusters pairwise
        pairwise_linked_labels = self._link_labels(labels_tmp, labels_shifted_tmp)
        unique_labels = self._omit_joint_clusters_in_unshifted_space(pairwise_linked_labels)

        # pairwise_linked_labels has shape (2, n).
        # The first row being the clusters in the shifted space, and the second in the unshifted space.

        # # Idea: Remove spikes which are clustered in neither space
        # noisy_in_both_spaces = (pairwise_linked_labels == self.NOISE_LABEL).all(axis=0)
        # if any(noisy_in_both_spaces):
        #     print("Remove things")
        #     pairwise_linked_labels = np.delete(pairwise_linked_labels, noisy_in_both_spaces, axis=1)

        # Idea: Find all those labels, which are clustered in the unshifted space.
        # joint_labels = self._get_splitted_clusterlabels(pairwise_linked_labels, pairwise_linked_labels[1])
        # if len(joint_labels):
        #     self._relabel_splitted_clusters(labels_tmp, labels_shifted_tmp, joint_labels, pairwise_linked_labels[1])




        for u_label in reversed(unique_labels):
        # for u_label in reversed(pairwise_linked_labels):
            # joint labels represent clusters, which are splitted in the shifted case.
            joint_labels = self._get_splitted_clusterlabels(pairwise_linked_labels, u_label)
            if u_label == self.NOISE_LABEL:
                continue
            if len(joint_labels):
                self._relabel_splitted_clusters(labels_tmp, labels_shifted_tmp, joint_labels, u_label)
            # A cluster that exists only in the shifted space
            # if u_label == self.NOISE_LABEL:
            #     # Retrieve all linked labels of such a case
            #     pairwise_linked_labels == self.NOISE_LABEL
            # print(u_label)

        return labels_tmp


    def _remove_noise_labels(self, data:np.ndarray, labels:np.ndarray):
        """
        data has shape: (n, 3) with n data points. Each column is aggregated by (time, x, y)
        """
        logger.info("Remove noise labels of the data.")
        return data[labels > self.NOISE_LABEL], labels[labels > self.NOISE_LABEL]


    def _link_labels(self, labels:np.ndarray, labels_shifted:np.ndarray):
        """
        Links the labels pairwise. And removes those links which only contains noise-labels.
        return:
            linked labels (shape is (l, 2) with l linked labels).
            The 0th index corresponds to the labels in the shifted space,
            the 1st index to the unshifted/original one.
        """
        pairwise_labels = np.unique(np.vstack([labels_shifted, labels]), axis=1)
        return pairwise_labels
        # Remove the links which are classified as noise in the shifted labels
        # pairwise_linked_labels = pairwise_labels[:, pairwise_labels[0] > self.NOISE_LABEL]

        logger.debug(f"Linked labels: {pairwise_labels}")
        logger.debug(f"Filtered links: {pairwise_linked_labels}")
        return pairwise_linked_labels


    def _get_splitted_clusterlabels(self, linked_labels:np.ndarray, clusterlabel_in_shifted_space:int)->tuple:
        """
        splitted_clusters specified by the identifier clusterlabel_in_shifted_space
        merged_label are non-noise label
        """
        # Reminder: linked_labels[0] is in the shifted space, linked_labels[1] is in the original space
        cluster_in_shifted_space = linked_labels[0] == clusterlabel_in_shifted_space
        splitted_clusters = linked_labels[1, cluster_in_shifted_space]
        joint_labels = splitted_clusters[splitted_clusters > self.NOISE_LABEL]
        logger.debug(f"Splitted clusters (Target label: {clusterlabel_in_shifted_space}): {joint_labels}")
        return joint_labels


    @staticmethod
    def _omit_joint_clusters_in_unshifted_space(pairwise_linked_labels:np.ndarray):
        """
        Remove these clusters which are fully catched in the original space and corresponds to splitted ones in the shifted space
        """
        # Count the occurences of the links among clusters.
        # A single appearance in the unshifted space corresponds to a joint cluster. It might get split in the shifted case.
        # Here, we omit this case.
        # Note: A cluster, which forms in the shifted space may include noisy points labeled in the unshifted space. This case is covered.
        unique_labels, label_count = np.unique(pairwise_linked_labels[0], return_counts=True)
        unique_labels = unique_labels[label_count >= 2]
        logger.debug(unique_labels)
        return unique_labels


    @staticmethod
    def _relabel_splitted_clusters(labels:np.ndarray, labels_shifted:np.ndarray, joint_labels, shifted_clusterlabel)->None:
        """
        Re-labels those data points, which were splitted (multiple labels, but one label in shifted space)
        & those which were noise in the original space but clustered in the shifted space.
        """
        target_label = joint_labels[0]
        for l in joint_labels:
            labels[labels == l] = target_label

        idx = labels_shifted == shifted_clusterlabel
        labels[idx] = target_label


    @staticmethod
    def _recenter_data(data:np.ndarray, nrows:int):
        """
        data has shape (n, 3)
        Shifts the data to a new center.
        """
        data_shifted = np.copy(data)
        data_shifted[:, 1:] = (data[:, 1:] + nrows / 2) % nrows
        assert np.all(data_shifted[:, 1:] <= nrows)
        return data_shifted
