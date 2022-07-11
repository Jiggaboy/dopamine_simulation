#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-07

@author: Hauke Wernecke

Creates plot according to the subspace angle module.
"""

import numpy as np
import matplotlib.pyplot as plt


def cumsum_variance(angle:object, tag:str)->None:
    plt.figure(f"cumsum_{tag}")
    plt.title("Cum. variance")
    plt.xlabel("PCs")
    plt.ylabel("Explained variance")
    for pca in angle.pcas:
        cumsum = pca.explained_variance_ratio_.cumsum()
        plt.plot(range(1, len(cumsum)+1), cumsum)
        
    
def angles(angle:object, tag:str)->None:
    plt.figure(f"angle_{tag}")
    plt.title("Angles between PCs")
    plt.xlabel("PCs")
    plt.ylabel("angle [°]")
    for a in angle.full_angles(*angle.pcas):
        plt.plot(range(1, len(a)+1), a, marker="*")