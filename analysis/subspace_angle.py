#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-06-30

@author: Hauke Wernecke

class SubspaceAngle measures the angle in a n-dimensional subspace which is analyzed by PCA.
The dimensionality correspond to the parameter n_components.
The measurement is performed between the baseline and the patch condition specified by the 'tag'.

"""

## TODO: Extend that one can give two pca objects to it.
## TODO: Extension: Compare two patches
## TODO: Use subset of neurons.

import numpy as np
from dataclasses import dataclass

from util import pickler as PIC
from util import functimer

from analysis.analysis import ratio_PCA


import cflogger
log = cflogger.getLogger()



@dataclass
class SubspaceAngle:
    config: object
    
    variance_threshold = .7
    
    
    @functimer
    def fit(self, tag:str, tag_ref:str, n_components:(int, float)=None, mask:np.ndarray=None):
        """
        Loads the rates of the tag (and tag_ref) and runs a PCA in order to calculate the angle of the subspaces.
        """
        # Set the default if n_components is not provided
        components = self.variance_threshold if n_components is None else n_components
        # Fit
        self.pseudo_fit(tag, tag_ref, components, mask)
        self.angle = self.angles_between_subspaces(*self.pcas)
        log.info(f"Angle after fit: {self.angle}")
        
        
    @functimer
    def pseudo_fit(self, tag:str, tag_ref:str=None, n_components:int=1, mask:np.ndarray=None):
        """
        Loads the rates of the tag (and tag_ref) and runs a PCA, BUT does not calculate the angle of the subspaces.
        """
        rates = self.load_rates(tag, tag_ref)
        rates = [rate[mask] for rate in rates] if mask is not None else rates
        
        self.pcas = self.disjoint_pca(*rates, components=n_components)
        
        min_components = np.max([pca.components_.shape[0] for pca in self.pcas])
        self.min_components_ = min_components
        log.info(f"Initial fit: {n_components}; Minimal components: {min_components}")
        
        for idx, pca in enumerate(self.pcas):
            if pca.components_.shape[0] < min_components:
                log.info(f"Rerun pca for indexed pca: {idx}")
                self.pcas[idx] = self.disjoint_pca(rates[idx], components=min_components)[0]
        
    
    def angles_between_subspaces(self, pca1, pca2, k:int=None):
        """Measures k-angles between the subspaces."""
        k = k if k is not None else self.min_components_
        print("Compnents", self.min_components_, k)
        _, s, _ = np.linalg.svd(np.dot(pca1.components_[:k+1], pca2.components_[:k+1].T), full_matrices=True)
        angles = np.arccos(s) * 180 / np.pi
        return angles
    
    
    def full_angles(self, pca1, pca2):
        return [self.angles_between_subspaces(pca1, pca2, k) for k in range(self.min_components_)]


    def load_rates(self, tag:str, tag_ref:str)->tuple:
        """
        Loads the baseline rate and the rate specified by tag.

        config:
            Determines the environment and the baseline tag.
        tag:
            Specifies the patch.
            Has previously to be specified in the config.
        """
        ref_rate = PIC.load_rate(postfix=tag_ref, skip_warmup=True, exc_only=True, sub_directory=self.config.sub_dir, config=self.config)
        patch_rate = PIC.load_rate(postfix=tag, skip_warmup=True, exc_only=True, sub_directory=self.config.sub_dir, config=self.config)

        return ref_rate, patch_rate


    def disjoint_pca(self, *data, components:int=3):
        """
        Runs a pca for each data set and returns the pca-objects.

        Each set in data: array-like of shape (n_samples, n_features)
        """
        return [ratio_PCA(d.T, n_components=components, plot=False) for d in data]
    
    
    def find_min_components(self, thr_variance:float=.7)->None:
        """
        Find the minimum of components required to explain thr_variance variance.
        
        thr_variance: default = .7
        """
        min_components = 1
        for pca in self.pcas:
            pcs_above_threshold = self._pcs_above_threshold(pca, thr_variance)
            min_components = max(min_components, pcs_above_threshold[0])
        self.min_component = min_components
    
    
    @staticmethod
    def _pcs_above_threshold(pca, threshold:float)->list:
        cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        return np.where(cum_explained_variance > threshold)[0]
        


def main():
    import matplotlib.pyplot as plt
    from params import PerlinConfig
    import dopamine as DP
    
    cfg = PerlinConfig()
    
    tag = "starter", "edge-activator"
    tag = "repeater", 
    MAX_components = 10
    thr_variance = .7
    
    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(43, 68), radius=10)
    LOCAL_NEURONS = DP.circular_patch(cfg.rows, center=(63, 34), radius=20)
    
    from universal import find_tags

    tags = find_tags(tag)
    angle = SubspaceAngle(cfg)
    
    tag = tags[1]
    
    def fixed_number():
        print(f"Run analysis with fixed number of components: {MAX_components}")
        angle.fit(tag, n_components=MAX_components, mask=LOCAL_NEURONS)
    
    def plain():
        print("Run analysis without specifying the number of components")
        angle.fit(tag, mask=LOCAL_NEURONS)
        
    def plot_cumsum_variance(angle, tag):
        plt.figure(f"cumsum_{tag}")
        for pca in angle.pcas:
            plt.plot(pca.explained_variance_ratio_.cumsum())
        
    def plot_angles(angle, tag):
        plt.figure(f"angle_{tag}")
        plt.plot(angle.angle)
    
    plain()
    
    plot_cumsum_variance(angle, tag)
    plot_angles(angle, tag)
    for a in angle.full_angles(*angle.pcas):
        plt.plot(a, marker="*")
    print(f"Component for 70% variance: {angle._pcs_above_threshold(angle.pcas[0], .7)})")
    print(f"Component for 90% variance: {angle._pcs_above_threshold(angle.pcas[0], .9)})")
    

    
    plt.show()
    

    


    


if __name__ == "__main__":
    main()
