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


            ####### TEST - START ######################################
            def find_target(angle) -> tuple:
                tile = np.pi / 8
                if np.isclose(angle, 0*tile, atol=1e-3):
                    return (2, 0)
                if np.isclose(angle, 1*tile, atol=1e-3):
                    return (2, 1)
                if np.isclose(angle, 2*tile, atol=1e-3):
                    return (2, 2)
                if np.isclose(angle, 3*tile, atol=1e-3):
                    return (1, 2)
                if np.isclose(angle, 4*tile, atol=1e-3):
                    return (0, 2)
                if np.isclose(angle, 5*tile, atol=1e-3):
                    return (-1, 2)
                if np.isclose(angle, 6*tile, atol=1e-3):
                    return (-2, 2)
                if np.isclose(angle, 7*tile, atol=1e-3):
                    return (-2, 1)
                if np.isclose(angle, -1*tile, atol=1e-3):
                    return (-2, 0)
                if np.isclose(angle, -2*tile, atol=1e-3):
                    return (-2, -1)
                if np.isclose(angle, -3*tile, atol=1e-3):
                    return (-2, -2)
                if np.isclose(angle, -4*tile, atol=1e-3):
                    return (-1, -2)
                if np.isclose(angle, -5*tile, atol=1e-3):
                    return (0, -2)
                if np.isclose(angle, -6*tile, atol=1e-3):
                    return (1, -2)
                if np.isclose(angle, -7*tile, atol=1e-3):
                    return (2, -2)
                if np.isclose(angle, -8*tile, atol=1e-3):
                    return (2, -1)

            def get_differences(target:np.ndarray):
                from lib.universal import get_coordinates
                coordinates = np.asarray(get_coordinates(5)) - 2
                coordinates = coordinates.reshape((5, 5, -1))
                return coordinates - target

            shift_matrix = conn.shift.reshape((config.rows, config.rows)).T
            d = shift_matrix[center]
            target = find_target(d)
            print(center, d, target)

            ds = np.roll(shift_matrix, -np.asarray(center)+2, axis=(0, 1))[:5, :5]
            print(ds)
            differences = get_differences(target)
            angles =
            quit()


            ####### TEST - END   ######################################

            #### PLAIN ################
            angle_x, angle_y = calculate_direction(conn.shift)
            directionality = np.linalg.norm([angle_x[patch].sum(), angle_y[patch].sum()])
            ###########################

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
