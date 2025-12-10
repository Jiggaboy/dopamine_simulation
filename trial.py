#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:


"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Hauke Wernecke'
__contact__ = 'hower@kth.se'
__version__ = '0.1'

#===============================================================================
# IMPORT STATEMENTS
#===============================================================================
from cflogger import logger

import numpy as np
import matplotlib.pyplot as plt
from plot.lib import add_colorbar
from matplotlib import rcParams
from params import config

import lib.pickler as PIC
from lib.connectivitymatrix import ConnectivityMatrix, CustomConnectivityMatrix
from plot.lib import plot_patch
import lib.universal as UNI
import plot.sequences as sq


degree_cmap = plt.cm.jet
min_degree = 575
max_degree = 850
min_degree = 0
max_degree = .075

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    force = UNI.yes_no("Force new connectivity matrix?", False)

    # conn = ConnectivityMatrix(config, force=force)
    conn = CustomConnectivityMatrix(config, force=force)

    for i, p in enumerate(config.PERCENTAGES):
        fig, ax = plt.subplots(num=f"{p}")
        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[p])

        duration_bs, sequence_count_bs = sq._get_count_and_duration(config, tags_by_seed[0], is_baseline=True)

        for s, tag_seeds in enumerate(tags_by_seed):
            duration, sequence_count = sq._get_count_and_duration(config, tag_seeds)

            duration_diff = (duration.mean() - duration_bs.mean()) / duration_bs.mean()
            sequence_diff = (sequence_count.mean() - sequence_count_bs.mean()) / sequence_count_bs.mean()

            magnitude = np.linalg.norm(duration_diff + sequence_diff)

            import lib.dopamine as DOP
            name = UNI.name_from_tag(tag_seeds[0])
            center = config.center_range[name]
            radius = UNI.radius_from_tag(tag_seeds[0])
            patch = DOP.circular_patch(config.rows, tuple(center), float(radius))

            angle_x, angle_y = calculate_direction(conn.shift)
            angle = np.arctan2(angle_y[patch].sum(), angle_x[patch].sum())
            directionality = np.linalg.norm([angle_x[patch].sum(), angle_y[patch].sum()])
            indegree = sq.get_indegree(config, tag_seeds)
            color = map_indegree_to_color(magnitude.mean())
            plt.scatter(directionality, indegree, color=color)
            plt.text(directionality, indegree, name, verticalalignment="bottom", horizontalalignment="center")
            # plt.scatter(indegree, magnitude.mean(), color=color)
        add_colorbar(ax, (min_degree, max_degree), cmap=degree_cmap)
        # break


#===============================================================================
# METHODS
#===============================================================================
def calculate_direction(x, **kwargs):
    u = np.cos(x)
    v = np.sin(x)
    return u, v


def map_indegree_to_color(indegree:float) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = degree_cmap((indegree - min_degree) / (max_degree - min_degree))
    return color





if __name__ == '__main__':
    main()
    plt.show()
