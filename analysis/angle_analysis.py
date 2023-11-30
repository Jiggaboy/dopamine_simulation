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

import cflogger
logger = cflogger.getLogger()


import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import namedtuple
# from collections.abc import Iterable

import lib.pickler as PIC
import lib.universal as UNI
import lib.dopamine as DOP

from analysis import AnalysisFrame
from analysis.lib import SubspaceAngle
from lib import functimer

from params import PerlinConfig

#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def analyze(config:object=None):
    cf = PerlinConfig() if config is None else config

    controls = cf.analysis.subspaceangle_controls

    angle = SubspaceAngle(cf)
    if controls.patch_against_baseline:
        pass
    if controls.patch_against_patch:
        pass
    if controls.baseline_against_baseline:
        pass


#===============================================================================
# CLASS
#===============================================================================


def get_mask(rows:int, center:tuple, radius:float)->np.ndarray:
    """
    Returns either a mask of indeces or None
    """
    try:
        logger.debug(f"Mask from rows: {rows}; center: {center}; radius: {radius}")
        return DOP.circular_patch(rows, center=center, radius=radius)
    except TypeError:
        logger.info("TypeError: No masked used!")


def subspace_angle_of_patch_with_baseline(config:object, center_tag:str, plot_angles:bool=True, plot_PC:bool=True)->None:
    """
    Global parameter:
        - ANGLE_RADIUS
    """
    angle = SubspaceAngle(config)
    center = config.get_center(center_tag) ##################################################### moved from the next for loop

    #####################################################
    N_COMPONENTS = 8
    from class_lib import AngleDumper
    angle_dumper = AngleDumper(
        tag=f"alignment_index_{center_tag}",
        center=center,
        radius=ANGLE_RADIUS,
        n_components=N_COMPONENTS
    )

    explained_variances = np.zeros(shape=(2, len(config.simulation_seeds), N_COMPONENTS))
    alignment_indexes = np.zeros(shape=(len(config.simulation_seeds), N_COMPONENTS))
    #####################################################

    for seed in config.simulation_seeds:
        tags = config.get_all_tags(center_tag, seeds=seed)
        bs_tag = config.baseline_tag(seed)
        logger.info(f"Found tags: {tags} with baseline {bs_tag}")
        for tag in tags:

            for r in ANGLE_RADIUS:
                mask = get_mask(config.rows, center=center, radius=r)
                angle.fit(tag, bs_tag, mask=mask) ##########################
                print(angle.min_components_)
                angle.fit(tag, bs_tag, mask=mask, n_components=N_COMPONENTS) ##########################
                t = f"{tag}_radius_{r}"
                if plot_angles:
                    _plot_angles.angles(angle, tag=t)
                if plot_PC:
                    for k in range(1, 5):
                        _plot_PC(config, *angle.pcas, k=k, patch=mask, figname=f"patch_bs_{t}_{seed}")


            ######################################################
            for i in range(2):
                explained_variances[i, seed] = angle.cumsum_variance(angle.pcas[i])
            #explained_variances[1, seed] = angle.cumsum_variance(angle.pca2)
            alignment_indexes[seed] = angle.full_alignment_indexes()


    fig = plt.figure()
    print(np.arange(1, N_COMPONENTS+1))
    print(explained_variances[0].mean(axis=0).shape)
    plt.errorbar(np.arange(1, N_COMPONENTS+1), explained_variances[0].mean(axis=0), yerr=explained_variances[0].std(axis=0))
    plt.errorbar(np.arange(1, N_COMPONENTS+1), explained_variances[1].mean(axis=0), yerr=explained_variances[1].std(axis=0))
    plt.errorbar(np.arange(1, N_COMPONENTS+1), alignment_indexes.mean(axis=0), yerr=alignment_indexes.std(axis=0))
    angle_dumper.explained_variances = explained_variances
    angle_dumper.alignment_indexes = alignment_indexes
    PIC.save_angle_dumper(angle_dumper, sub_directory=config.sub_dir)
    ######################################################


def subspace_angle_of_patch_with_patch(config:object, center_tag:str, plot_angles:bool=True, plot_PC:bool=True)->None:
    """
    Global parameter:
        - ANGLE_RADIUS
    """
    angle = SubspaceAngle(config)
    tags = config.get_all_tags(center_tag)
    logger.info(f"Tags: {center_tag} -> {tags}")
    for i, tag in enumerate(tags):
        center = config.get_center(r_tag)
        for j, tag_ref in enumerate(tags):
            # Skip identical simulations
            if i >= j:
                continue
            for r in ANGLE_RADIUS:
                mask = get_mask(config.rows, center=center, radius=r)
                _identifier = "_radius_" + str(r)

                logger.info(f"FIT: {tag} vs {tag_ref}")
                angle.fit(tag, tag_ref, mask=mask)
                t_mixed = tag + "_" + tag_ref + _identifier
                if plot_angles:
                    _plot_angles.angles(angle, tag=t_mixed)
                if plot_PC:
                    _plot_PC(config, angle.pcas[0], mask, figname=f"reference_{t_ref}_{seed}")
                    _plot_PC(config, angle.pcas[1], mask, figname=f"target_{t}_{seed}")


def subspace_angle_of_baseline_with_baseline(config:object, center_tag:str, plot_angles:bool=True, plot_PC:bool=True)->None:
    """
    Global parameter:
        - ANGLE_RADIUS
    """
    angle = SubspaceAngle(config)
    center = config.get_center(center_tag)
    tags = config.baseline_tags
    logger.info(f"Baseline tags to compare: {tags}")
    for i, tag in enumerate(tags):
        for j, tag_ref in enumerate(tags):
            # Skip identical simulations
            if i >= j:
                continue
            for r in ANGLE_RADIUS:
                mask = get_mask(config.rows, center=center, radius=r)
                _identifier = f"_radius_{r}_{center}"

                logger.info(f"FIT: {tag} vs {tag_ref}")
                angle.fit(tag, tag_ref, mask=mask)
                t_mixed = tag + "_" + tag_ref + _identifier
                if plot_angles:
                    _plot_angles.angles(angle, tag=t_mixed)
                if plot_PC:
                    _plot_PC(config, angle.pcas[0], mask, figname=f"reference_{t_ref}_{seed}")
                    _plot_PC(config, angle.pcas[1], mask, figname=f"target_{t}_{seed}")


def subspace_angle(config:object, plain_tags:list, plot_angles:bool=True, plot_PC:bool=True)->None:
    """
    plain_tags: just the name of the patches, like 'repeater', 'linker', ....
    """
    angle = SubspaceAngle(config)

    from class_lib import AngleDumper

    for r_tag in plain_tags:
        if PATCH_CROSS_BASELINE:
            subspace_angle_of_patch_with_baseline(config, r_tag, plot_angles, plot_PC)

        if CROSS_ANGLES:
            subspace_angle_of_patch_with_patch(config, r_tag, plot_angles, plot_PC)

        if CROSS_BASELINES:
            #subspace_angle_of_baseline_with_baseline(config, r_tag, plot_angles, plot_PC)


            N_COMPONENTS = 8

            center = config.get_center(r_tag)
            tags = config.baseline_tags
            logger.info(f"Baseline tags to compare: {tags}")

            angle_dumper = AngleDumper(
                tag=f"angles_across_baselines_{center}",
                center=center,
                radius=ANGLE_RADIUS,
                n_components=N_COMPONENTS
            )
            for idx, radius in enumerate(ANGLE_RADIUS):
                mask = get_mask(config.rows, center=center, radius=radius)
                pooled_angles = init_triangular_matrix(N_COMPONENTS)

                for i, tag_ref in enumerate(tags):
                    for j, tag in enumerate(tags):
                        if i >= j:
                            continue

                        logger.info(f"BASELNIE comparison: {tag} with {tag_ref}")
                        angle.fit(tag, tag_ref, mask=mask, n_components=N_COMPONENTS)
                        for k in range(N_COMPONENTS):
                            pooled_angles[k].append(angle.angles_between_subspaces(k=k))
                angle_dumper.angles[radius] = pooled_angles
            PIC.save_angle_dumper(angle_dumper, sub_directory=config.sub_dir)



def init_triangular_matrix(n_elements:int):
    """Initialize a vector of length n_elements with empty lists."""
    matrix = np.empty(n_elements, dtype=object)
    for i in range(n_elements):
        matrix[i] = []
    return matrix





if __name__ == '__main__':
    analyze()
