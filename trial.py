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


from plot.lib import plot_patch
import lib.universal as UNI
import plot.sequences as sq

from lib.neuralhdf5 import NeuralHdf5, default_filename

degree_cmap = plt.cm.jet
min_degree = 575
max_degree = 850
min_degree = 0
max_degree = .25

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    side = 2
    with NeuralHdf5(default_filename, "a", config=config) as file:
        shift_matrix = file.shift.reshape((config.rows, config.rows)).T
        
        durations_bs       = np.zeros(len(config.baseline_tags))
        sequence_counts_bs = np.zeros(len(config.baseline_tags))
        
        for t, tag in enumerate(config.baseline_tags):
            tmp, sequence_counts_bs[t] = file.get_sequence_duration_and_count(tag, is_baseline=True)
            durations_bs[t] = tmp.mean()
            
            
        durations        = np.zeros((len(config.PERCENTAGES), len(config.center_range),config.drive.seeds.size))
        sequence_counts  = np.zeros((len(config.PERCENTAGES), len(config.center_range), config.drive.seeds.size))
        directionalities = np.zeros((len(config.PERCENTAGES), len(config.center_range),  2*side+1, 2*side+1))
        directionalities_per_neuron = np.zeros((len(config.center_range), config.drive.seeds.size, config.AMOUNT_NEURONS[0], 2*side+1, 2*side+1))
        
        indegree = file.get_indegree()
        indegrees = np.zeros((len(config.center_range), config.drive.seeds.size))
        
        for i, percent in enumerate(config.PERCENTAGES):
            tags_by_seed = config.get_all_tags(seeds="all", weight_change=[percent])
            for t, tags in enumerate(tags_by_seed):
                for j, tag in enumerate(tags):
                    # file.reset_sequence_duration_and_count(tag)
                    tmp, sequence_counts[i, t, j] = file.get_sequence_duration_and_count(tag)
                    durations[i, t, j] = tmp.mean()
                    
                    ### TEST - START: DIRECTIONALITY PER NEURON #################################################
                    if i > 0:
                        continue
                    name = UNI.name_from_tag(tag)
                    center = config.center_range[name]
                    radius = UNI.radius_from_tag(tag)
                    _, seed = UNI.split_seed_from_tag(tag)
                    import lib.dopamine as DOP
                    from lib.lcrn_network import targets_to_grid
                    patch = DOP.circular_patch(config.rows, np.asarray(center), float(radius))
                    neurons = UNI.get_neurons_from_patch(patch, config.AMOUNT_NEURONS[0], int(seed)+1)
                    
                    
                    indegrees[t, j] = indegree.ravel()[neurons].mean() * config.synapse.weight
                    
                    
                    for n, neuron in enumerate(neurons):
                        print(neuron)
                        idx2 = np.floor(neuron / config.rows).astype(int), neuron % config.rows
                        print(idx2)
                        
                        for pi, p in enumerate(range(-side, side+1)):
                            for qi, q in enumerate(range(-side, side+1)):
                                center_tmp = np.asarray(idx2).astype(int) + np.asarray([p, q])
                                center_tmp %= config.rows
                                d = shift_matrix[tuple(center_tmp)]
                                target = find_target(d)
            
                                # ds are the actual part of the shift matrix
                                ds = np.roll(shift_matrix, -np.asarray(center_tmp) + 2, axis=(0, 1))[:5, :5]
            
                                differences = get_differences(target).astype(float)
                                angles = np.arctan2(-differences[:, :, 1], -differences[:, :, 0])
            
                                directionality = (ds.T - angles)
                                directionality = (directionality + np.pi) % (2*np.pi) - np.pi
                                mask = np.asarray(target) + [2, 2]
                                directionality[tuple(mask[::1])] = 0 # Filter any artifact of the target location
                                directionality = np.abs(directionality).mean()
                                directionalities_per_neuron[t, j, n, pi, qi] = directionality
                            
                    ### TEST - STOP : DIRECTIONALITY PER NEURON #################################################
                
                name = UNI.name_from_tag(tags[0])
                center = config.center_range[name]
                radius = UNI.radius_from_tag(tags[0])
                
            
                for pi, p in enumerate(range(-side, side+1)):
                    for qi, q in enumerate(range(-side, side+1)):
                        center_tmp = center + np.asarray([p, q])
                        center_tmp %= config.rows
                        d = shift_matrix[tuple(center_tmp)]
                        target = find_target(d)
    
                        # ds are the actual part of the shift matrix
                        ds = np.roll(shift_matrix, -np.asarray(center_tmp) + 2, axis=(0, 1))[:5, :5]
    
                        differences = get_differences(target).astype(float)
                        angles = np.arctan2(-differences[:, :, 1], -differences[:, :, 0])
    
                        directionality = (ds.T - angles)
                        directionality = (directionality + np.pi) % (2*np.pi) - np.pi
                        mask = np.asarray(target) + [2, 2]
                        directionality[tuple(mask[::1])] = 0 # Filter any artifact of the target location
                        directionality = np.abs(directionality).mean()
    
                        if i == -1:
                            plt.figure(f"{name} -{p}-{q} ({p}) - angles")
                            plt.imshow(angles, origin="lower")
                            plt.colorbar()
                            plt.figure(f"{name} -{p}-{q}({p}) - ds.T")
                            plt.imshow(ds.T, origin="lower")
                            plt.colorbar()
                            plt.figure(f"{name} -{p}-{q}({p}) - ds.T-angles")
                            plt.imshow(ds.T-angles, origin="lower")
                            plt.colorbar()
                        directionalities[i, t, pi, qi] = directionality
    


        for i, percent in enumerate(config.PERCENTAGES):
            fig, ax = plt.subplots(num=f"percentage: {percent}")
            ax.set_xlabel("coherence score")
            ax.set_ylabel("mean indegree")
            fig.suptitle("Color indicates impact of the patch")
            add_colorbar(ax, (min_degree, max_degree), cmap=degree_cmap)
            
            
            tags_by_seed = config.get_all_tags(seeds="all", weight_change=[percent])
            for t, tags in enumerate(tags_by_seed):
                for j, tag in enumerate(tags):
                    
                    name = UNI.name_from_tag(tags[0])
                    center = config.center_range[name]
            
            # for c, (name, center) in enumerate(config.center_range.items()):
            #     print(c, center)
                
                    duration_diff = ((durations[i, t] - durations_bs) / durations_bs).mean()
                    sequence_diff = ((sequence_counts[i, t] - sequence_counts_bs) / sequence_counts_bs).mean()
                    
                    magnitude = np.linalg.norm([duration_diff, sequence_diff])
                    
                    
                    # center = tuple(int(c) for c in center)
                    # indegree = file.get_indegree(center)
                    indegree = indegrees[t, j]
                    
                    color = map_indegree_to_color(magnitude)
                    duration_diff = (durations[i, t, j] - durations_bs) / durations_bs
                    sequence_diff = (sequence_counts[i, t, j] - sequence_counts_bs) / sequence_counts_bs
                    
                    magnitude = np.linalg.norm([duration_diff, sequence_diff])
                    color = map_indegree_to_color(magnitude)
                    
                    direction = directionalities_per_neuron[t, j].mean()
                    # direction = directionalities[i, c].mean()
                    
                    ax.scatter(direction, indegree, color=color)
                    ax.plot([direction, directionalities_per_neuron[t].mean()],
                            [indegree, indegrees[t].mean()],
                            zorder=-1, color="grey")
                ax.text(directionalities_per_neuron[t].mean(), indegrees[t].mean(), name, verticalalignment="bottom", horizontalalignment="center")
                

    return

    magnitude_dict = {}
    fig, ax = plt.subplots(num="merged")
    for i, percent in enumerate(config.PERCENTAGES):
        # fig, ax = plt.subplots(num=f"{percent}")
        plt.xlabel("coherence score")
        plt.ylabel("mean indegree")
        plt.title("Color indicates impact of the patch")
        tags_by_seed = config.get_all_tags(seeds="all", weight_change=[percent])


        for s, tag_seeds in enumerate(tags_by_seed):
            duration_bs, sequence_count_bs = sq._get_count_and_duration(config, tag_seeds, is_baseline=True)
            duration, sequence_count = sq._get_count_and_duration(config, tag_seeds)

            duration_diff = (duration.mean() - duration_bs.mean()) / duration_bs.mean()
            sequence_diff = (sequence_count.mean() - sequence_count_bs.mean()) / sequence_count_bs.mean()

            magnitude = np.linalg.norm([duration_diff, sequence_diff])

            import lib.dopamine as DOP
            name = UNI.name_from_tag(tag_seeds[0])
            center = config.center_range[name]
            radius = UNI.radius_from_tag(tag_seeds[0])
            patch = DOP.circular_patch(config.rows, tuple(center), float(radius))


            ####### TEST - START ######################################
            shift_matrix = conn.shift.reshape((config.rows, config.rows)).T

            # if name != "loc-7":
            #     continue

            side = 4
            directionalities = np.zeros((2*side+1, 2*side+1))
            for pi, p in enumerate(range(-side, side+1)):
                for qi, q in enumerate(range(-side, side+1)):
                    center_tmp = center + np.asarray([p, q])
                    center_tmp %= config.rows
                    d = shift_matrix[tuple(center_tmp)]
                    target = find_target(d)

                    # ds are the actual part of the shift matrix
                    ds = np.roll(shift_matrix, -np.asarray(center_tmp)+2, axis=(0, 1))[:5, :5]

                    differences = get_differences(target).astype(float)
                    angles = np.arctan2(-differences[:, :, 1], -differences[:, :, 0])

                    directionality = (ds.T - angles)
                    directionality = (directionality + np.pi) % (2*np.pi) - np.pi
                    mask = np.asarray(target) + [2, 2]
                    directionality[tuple(mask[::1])] = 0 # Filter any artifact of the target location
                    directionality = np.abs(directionality).mean()

                    if i == -1:
                        plt.figure(f"{name} -{p}-{q} ({p}) - angles")
                        plt.imshow(angles, origin="lower")
                        plt.colorbar()
                        plt.figure(f"{name} -{p}-{q}({p}) - ds.T")
                        plt.imshow(ds.T, origin="lower")
                        plt.colorbar()
                        plt.figure(f"{name} -{p}-{q}({p}) - ds.T-angles")
                        plt.imshow(ds.T-angles, origin="lower")
                        plt.colorbar()
                    directionalities[pi, qi] = directionality
                    indegree = sq.get_indegree(config, tag_seeds)
                    # color = map_indegree_to_color(magnitude.mean())
                    # ax.scatter(directionality, indegree, color=color)

            ####### TEST - END   ######################################

            #### PLAIN ################
            from figure_generator.in_out_degree import calculate_direction
            # angle_x, angle_y = calculate_direction(conn.shift)
            # directionality = np.linalg.norm([angle_x[patch].sum(), angle_y[patch].sum()])
            ###########################
            magnitude_dict[name] = magnitude.mean() if magnitude.mean() > magnitude_dict.get(name, 0) else magnitude_dict[name]

            indegree = sq.get_indegree(config, tag_seeds)
            # color = map_indegree_to_color(magnitude.mean())
            color = map_indegree_to_color(magnitude_dict[name])
            ax.scatter(directionalities.mean(), indegree, color=color, marker="*")
            # ax.scatter(directionalities.mean(), indegree, color=magnitude_dict[name], marker="*")
            ax.text(directionalities.mean(), indegree, name, verticalalignment="bottom", horizontalalignment="center")
            # plt.scatter(indegree, magnitude.mean(), color=color)
        add_colorbar(ax, (min_degree, max_degree), cmap=degree_cmap)


        # break



#===============================================================================
# METHODS
#===============================================================================
def map_indegree_to_color(indegree:float) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = degree_cmap((indegree - min_degree) / (max_degree - min_degree))
    return color

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
        return (2, -1)
    if np.isclose(angle, -2*tile, atol=1e-3):
        return (2, -2)
    if np.isclose(angle, -3*tile, atol=1e-3):
        return (1, -2)
    if np.isclose(angle, -4*tile, atol=1e-3):
        return (0, -2)
    if np.isclose(angle, -5*tile, atol=1e-3):
        return (-1, -2)
    if np.isclose(angle, -6*tile, atol=1e-3):
        return (-2, -2)
    if np.isclose(angle, -7*tile, atol=1e-3):
        return (-2, -1)
    if np.isclose(angle, -8*tile, atol=1e-3):
        return (-2, 0)

def get_differences(target:np.ndarray):
    positions = np.arange(-2, 2+1, step=1)
    x, y = np.meshgrid(positions, positions)
    coordinates = np.zeros((5, 5, 2))
    coordinates[:, :, 0] = x
    coordinates[:, :, 1] = y
    return coordinates - target








if __name__ == '__main__':
    main()
    plt.show()
