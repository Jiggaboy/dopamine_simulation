#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

class SubspaceAngle measures the angle in a n-dimensional subspace which is analyzed by PCA.
The dimensionality correspond to the parameter n_components.
The measurement is performed between two conditions specified by 'tag' and 'tag_ref'.

Note: The first pca in self.pcas is the baseline/tag_ref pca.

Methods:
    - fit(self, tag:str, tag_ref:str, n_components:(int, float)=None, mask:np.ndarray=None)
        Fits the data and determines the n_components. Mask is optional to reduce the number of features/neurons.
    - cumsum_variance(pca)
        Returns the cumulated explained variance for a given pca-object. PCA objects are stored in SubspaceAngle.pcas
        Usage:
            for pca in angle.pcas:
                plt.plot(angle.cumsum_variance(pca))
    - full_angles()
        Returns a list of angles. The angles are betwenn the first n PCs. Maximum number of angles is determined by n_components.
"""

## TODO: Extend that one can give two pca objects to it.


import cflogger
log = cflogger.getLogger()

import numpy as np
from dataclasses import dataclass

from lib import pickler as PIC
from lib import functimer

from analysis.pca import PCA, sum_variances


@dataclass
class SubspaceAngle:
    config: object

    variance_threshold = .7

    def cumsum_variance(self, pca:object)->np.ndarray:
        return pca.explained_variance_ratio_.cumsum()[:self.min_components_]


    @functimer
    def fit(self, tag:str, tag_ref:str, n_components:(int, float)=None, mask:np.ndarray=None):
        """
        Loads the rates of the tag (and tag_ref) and runs a PCA in order to calculate the angle of the subspaces.
        """
        # Fit
        self.pseudo_fit(tag, tag_ref, mask)
        # Set the default if n_components is not provided
        self._get_min_components(n_components)
        self.angle = self.angles_between_subspaces(*self.pcas)
        log.info(f"Angle after fit: {self.angle}")


    @functimer
    def pseudo_fit(self, tag:str, tag_ref:str=None, mask:np.ndarray=None):
        """
        Loads the rates of the tag (and tag_ref) and runs a PCA, BUT does not calculate the angle of the subspaces.
        """
        self.pcas = []

        for t in (tag_ref, tag):
            rate = self.load_rate(t)
            log.debug(f"Shape of rate: {rate.shape}, mask is: {mask}")
            if mask is not None:
                log.info("Force local PCA...")
                log.debug(f"Shape of rate: {rate[mask].shape}")
                self.pcas.append(PCA(rate[mask].T, force=True))
            else:
                self.pcas.append(PCA(rate.T, filename=t))


    def _get_min_components(self, n_components:(int, float)):
        """
        Determines the number of components to use.
        If n_components is a float, the no. of PCs that explain at least n_components% of the variance in the data.
        """
        components = self.variance_threshold if n_components is None else n_components
        if components >= 1:
            self.min_components_ = components
        elif components < 1.0:
            min_component = -1
            for pca in self.pcas:
                _, explained_var_ratio = sum_variances(pca.explained_variance_ratio_)
                crossing = np.argmax(explained_var_ratio > components) + 1
                min_component = max(min_component, crossing)
            self.min_components_ = min_component


    def angles_between_subspaces(self, pca1=None, pca2=None, k:int=None):
        """Measures k-angles between the subspaces."""
        k = k if k is not None else self.min_components_
        pca1, pca2 = self.pcas if pca2 is None else (pca1, pca2)
        log.debug("Components k: ", k)
        _, s, _ = np.linalg.svd(np.dot(pca1.components_[:k+1], pca2.components_[:k+1].T), full_matrices=True)
        angles = np.arccos(s) * 180 / np.pi
        return angles


    def alignment_index(self, pca1=None, pca2=None, PCs:int=None)->float:
        """
        Defined in Elsayed et al. 2016
        """
        PCs = PCs if PCs is not None else self.min_components_
        pca1, pca2 = self.pcas if pca2 is None else (pca1, pca2)

        cov = pca1.get_covariance()
        subspace_modes = pca2.components_[:PCs+1].T

        numerator = subspace_modes.T.dot(cov).dot(subspace_modes)
        trace = np.trace(numerator)

        # euivalent: sv = np.linalg.svd(cov, compute_uv=False, hermitian=True)[:PCs+1]
        sv = pca1.singular_values_[:PCs+1]**2 / (pca1.n_samples_ - 1)
        log.debug(f"Numerator: {np.trace(numerator)}")
        log.debug(f"Denominator: {sv.sum()}")
        return np.trace(numerator) / sv.sum()


    def full_angles(self, pca1=None, pca2=None)->list:
        """
        Returns a list of angles. The angles are betwenn the first n PCs. Maximum number of angles is determined by n_components.

        Takes two pca objects or takes the pcas of the self-object.
        """
        pca1, pca2 = self.pcas if pca2 is None else (pca1, pca2)
        return [self.angles_between_subspaces(pca1, pca2, k) for k in range(self.min_components_)]


    def full_alignment_indexes(self, pca1=None, pca2=None)->list:
        """
        Returns a list of alignment indexes. The alignment indexes are betwenn the first n PCs. Maximum number of angles is determined by n_components.

        Takes two pca objects or takes the pcas of the self-object.
        """
        pca1, pca2 = self.pcas if pca2 is None else (pca1, pca2)
        log.debug(f"Get alignment index for {self.min_components_} PCs.")
        return [self.alignment_index(pca1, pca2, k) for k in range(self.min_components_)]



    def load_rate(self, tag:str)->np.ndarray:
        """
        Loads the rate specified by tag.

        config:
            Determines the environment and the baseline tag.
        tag:
            Specifies the patch.
            Has previously to be specified in the config.
        """
        return PIC.load_rate(postfix=tag, skip_warmup=True, exc_only=True, sub_directory=self.config.sub_dir, config=self.config)


    @staticmethod
    def _pcs_above_threshold(pca, threshold:float)->list:
        cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        return np.where(cum_explained_variance > threshold)[0]



def main():
    import matplotlib.pyplot as plt
    from params import PerlinConfig
    import lib.dopamine as DP

    cfg = PerlinConfig()

    tag = "starter", "edge-activator"
    tag = "out-activator",
    tag = "repeater",
    MAX_components = 10
    thr_variance = .7
    thr_variance = .9

    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(17, 34), radius=10)
    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=cfg.get_center(tag[0]), radius=14)
    # LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(59, 34), radius=14)


    tags = cfg.get_all_tags(tag)
    bs_tag = cfg.baseline_tags[0]
    angle = SubspaceAngle(cfg)

    tag = tags[1]
    # tag = cfg.baseline_tags[1]

    def fixed_number():
        print(f"Run analysis with fixed number of components: {MAX_components}")
        angle.fit(tag, bs_tag, n_components=MAX_components, mask=LOCAL_NEURONS)

    def plain():
        print("Run analysis without specifying the number of components")
        angle.fit(tag, bs_tag, n_components=thr_variance,  mask=LOCAL_NEURONS)

    def plot_cumsum_variance(angle, tag):
        plt.figure(f"cumsum_{tag}")
        for pca in angle.pcas:
            plt.plot(angle.cumsum_variance(pca))

    def plot_angles(angle, tag):
        plt.figure(f"angle_{tag}")
        print(f"angles: {angle.angle}")
        print(f"min_comp: {angle.min_components_}")
        for a in angle.full_angles():
            plt.plot(a, marker="*")

    fixed_number()
    from plot import angles as _plot_angles
    _plot_angles.angles(angle, tag, plot=False)
    print(f"Component for 70% variance: {angle._pcs_above_threshold(angle.pcas[0], .7)})")
    print(f"Component for 90% variance: {angle._pcs_above_threshold(angle.pcas[0], .9)})")


    print("Alignment indexes:", angle.full_alignment_indexes())
    return
    cov = angle.pcas[0].get_covariance()
    print("Singular values of sklearn.decomposition.PCA:")
    print(angle.pcas[0].singular_values_[:10])
    print("Explained variance of sklearn.decomposition.PCA:")
    print(angle.pcas[0].explained_variance_[:10])
    print("Compute singular values using np.linalg.svd:")
    print(np.linalg.svd(cov, compute_uv=False)[:10])
    print(angle.pcas[0].singular_values_[:10]**2 / np.linalg.svd(cov, compute_uv=False)[:10])

    # plt.show()








if __name__ == "__main__":
    main()
