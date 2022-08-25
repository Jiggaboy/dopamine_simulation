#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-07

@author: Hauke Wernecke

Creates plot according to the subspace angle module.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _cumsum_variance(axis, angle:object)->None:
    axis.set_title("Cum. variance")
    axis.set_xlabel("PCs")
    axis.set_ylabel("Explained variance")
    for pca in angle.pcas:
        cumsum = angle.cumsum_variance(pca)
        axis.plot(range(1, len(cumsum)+1), cumsum)
    axis.legend(["baseline", "with patch"])
        

def _angles(axis, angle:object)->None:
    axis.set_title("Angles between PCs")
    axis.set_xlabel("PCs")
    axis.set_ylabel("angle [°]")
    for a in angle.full_angles(*angle.pcas):
        axis.plot(range(1, len(a)+1), a, marker="*")
 
    
def angles(angle:object, tag:str)->None:
    figname = f"angle_{tag}"
    fig, (axis_angle, axis_cumsum_var) = plt.subplots(nrows=2, num=figname, figsize=(6, 8))
    _angles(axis_angle, angle)
    _cumsum_variance(axis_cumsum_var, angle)
    plt.savefig(os.path.join("figures", "angle", figname) + ".svg")