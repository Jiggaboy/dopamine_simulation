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
max_degree = .2

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    side = 0
    with NeuralHdf5(default_filename, "a", config=config) as file:
        shift_matrix = file.shift.reshape((config.rows, config.rows))
        
        durations_bs       = np.zeros(len(config.baseline_tags))
        sequence_counts_bs = np.zeros(len(config.baseline_tags))
        
        for t, tag in enumerate(config.baseline_tags):
            # file.reset_sequence_duration_and_count(tag, is_baseline=True)
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
                    file.reset_sequence_duration_and_count(tag)
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
                    patch = DOP.circular_patch(config.rows, np.asarray(center), float(radius))
                    neurons = UNI.get_neurons_from_patch(patch, config.AMOUNT_NEURONS[0], int(1e4*center[0] + 1e2*center[1] + int(seed)+1))

                    indegrees[t, j] = indegree.ravel()[neurons].mean() * config.synapse.weight                    
                    
                    for n, neuron in enumerate(neurons):
                        idx2 = neuron % config.rows, np.floor(neuron / config.rows).astype(int)
                        
                        for pi, p in enumerate(range(-side, side+1)):
                            for qi, q in enumerate(range(-side, side+1)):
                                center_tmp = np.asarray(idx2).astype(int) + np.asarray([p, q])
                                center_tmp %= config.rows
                                directionalities_per_neuron[t, j, n, pi, qi] = get_coherence_score(shift_matrix, center_tmp)
                            
                    ### TEST - STOP : DIRECTIONALITY PER NEURON #################################################
                
                name = UNI.name_from_tag(tags[0])
                center = config.center_range[name]
                radius = UNI.radius_from_tag(tags[0])
                
                for pi, p in enumerate(range(-side, side+1)):
                    for qi, q in enumerate(range(-side, side+1)):
                        center_tmp = center + np.asarray([p, q])
                        center_tmp %= config.rows
                        directionalities[i, t, pi, qi] = get_coherence_score(shift_matrix, center_tmp)
    


        for i, percent in enumerate(config.PERCENTAGES):
            fig, ax = plt.subplots(num=f"percentage: {percent}")
            ax.set_xlabel("coherence score")
            ax.set_ylabel("mean indegree")
            fig.suptitle("Color indicates impact of the patch")
            add_colorbar(ax, (min_degree, max_degree), cmap=degree_cmap)
            
            
            tags_by_seed = config.get_all_tags(seeds="all", weight_change=[percent])
            for t, tags in enumerate(tags_by_seed):
                for j, tag in enumerate(tags):
                    # ### OPTION A: Coherence score by center of patch
                    # direction = directionalities[i, t].mean()
                    # # Get magnitude
                    # duration_diff = (durations[i, t] - durations_bs) / durations_bs
                    # sequence_diff = (sequence_counts[i, t] - sequence_counts_bs) / sequence_counts_bs
                    # magnitude = np.linalg.norm([duration_diff.mean(), sequence_diff.mean()])
                    # # Mapping
                    # color = map_indegree_to_color(magnitude)
                    # # Indegree by patch
                    # name = UNI.name_from_tag(tags[0])
                    # center = config.center_range[name]
                    # center = tuple(int(c) for c in center)
                    # radius  = UNI.radius_from_tag(tag)  
                    # indegree = file.get_indegree(center, radius)
                                        
                    
                    ### OPTION B: Coherence score by modulated neurons
                    name = UNI.name_from_tag(tags[0])
                    direction = directionalities_per_neuron[t, j].mean()
                    # Get magnitude
                    duration_diff = ((durations[i, t, j] - durations_bs[j]) / durations_bs[j])
                    sequence_diff = ((sequence_counts[i, t, j] - sequence_counts_bs[j]) / sequence_counts_bs[j])
                    magnitude = np.linalg.norm([duration_diff, sequence_diff])
                    # Mapping
                    # TODO: Rename method!!!
                    color = map_indegree_to_color(magnitude)
                    # Indegree by modulated neurons
                    indegree = indegrees[t, j]
                                        
                    ax.scatter(direction, indegree, color=color)
                    ax.plot([direction, directionalities_per_neuron[t].mean()],
                            [indegree, indegrees[t].mean()],
                            zorder=-1, color="lightgrey")
                ax.text(directionalities_per_neuron[t].mean(), indegrees[t].mean(), name, verticalalignment="bottom", horizontalalignment="center")


#===============================================================================
# METHODS
#===============================================================================
def map_indegree_to_color(indegree:float) -> float:
    indegree = min_degree if indegree < min_degree else indegree
    indegree = max_degree if indegree > max_degree else indegree
    color = degree_cmap((indegree - min_degree) / (max_degree - min_degree))
    return color


def get_coherence_score(shift_matrix:np.ndarray, center:tuple):
    angle = shift_matrix.T[tuple(center)]     # Get the shift of the center
    target = find_target(angle)             # Get the target in the 5x5 matrix
    
    angles = np.roll(shift_matrix, -np.asarray(center) + 2, axis=(1, 0))[:5, :5] # axis=(1, 0) for transposing purposes
    # ##### TEST: START ################################
    # fig, axes = plt.subplots(ncols=3)
    # angles_plain = np.roll(shift_matrix, -np.asarray(center) + 2, axis=(0, 1))[:5, :5] 
    # axes[0].imshow(angles_plain.T, origin="lower", cmap=degree_cmap, vmin=-np.pi, vmax=np.pi)
    # axes[1].imshow(angles, origin="lower", cmap=degree_cmap, vmin=-np.pi, vmax=np.pi)
    # axes[2].imshow(angles.T, origin="lower", cmap=degree_cmap, vmin=-np.pi, vmax=np.pi)
    # ##### TEST: STOP ################################
    
    differences_to_target = get_differences(target).astype(float)
    angles_to_target = np.arctan2(-differences_to_target[:, :, 1], -differences_to_target[:, :, 0]) # -sign due to imagination problems...    

    alignment = (angles - angles_to_target)
    # Normalization
    alignment = (alignment + np.pi) % (2*np.pi) - np.pi
    
    # ##### TEST: START ################################
    # fig, axes = plt.subplots(ncols=3)
    # axes[0].imshow(angles_to_target, origin="lower", cmap=degree_cmap, vmin=-np.pi, vmax=np.pi)
    # axes[1].imshow(alignment, origin="lower", cmap=degree_cmap, vmin=-np.pi, vmax=np.pi)
    # ##### TEST: STOP ################################
            
    mask = np.asarray(target) + [2, 2]
    alignment[tuple(mask[::-1])] = 0 # Filter any artifact of the target location
    
    # ##### TEST: START ################################
    # axes[2].imshow(alignment, origin="lower", cmap=degree_cmap, vmin=-np.pi, vmax=np.pi)
    # plt.show(block=False)
    # ##### TEST: STOP ################################
    return np.abs(alignment).mean()


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
