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

from params import config

import lib.universal as UNI
from lib.connectivitymatrix import ConnectivityMatrix
from analysis.sequence_correlation import SequenceCorrelator
import analysis.dbscan_sequences as dbs
from analysis.activity import _average_rate
import lib.dopamine as DOP

from lib.neuralhdf5 import NeuralHdf5
from lib.neuralhdf5 import default_filename, connectivity_tag, shift_tag, indegree_tag, matrix_tag
from lib.neuralhdf5 import activity_tag, avg_rate_tag, spikes_tag, patch_tag, labels_tag
from lib.neuralhdf5 import baseline_tag as baseline

# Cluster activity

specific_tag = "repeat"
specific_tag = None


import lib.pickler as PIC
def average_rate(tag, force:bool=False, **save_params):
    """Averages the rates of the given tags. Saves the averaged rates."""
    # if not force and PIC.datafile_exists(PIC.load_average_rate(tag, dry_run=True), **save_params):
    #     return

    try:
        rate = PIC.load_rate(tag, exc_only=True, **save_params)
    except FileNotFoundError:
        return
    
    avgRate = rate.mean(axis=1)
    PIC.save_avg_rate(avgRate, tag, **save_params)
    return avgRate
#===============================================================================
# MAIN METHOD
#===============================================================================
def main(config):
    force_patch = UNI.yes_no("Force clustering for patch simulations?", False)
    force_baseline = UNI.yes_no("Force baseline clustering?", False)
    force_patch_averaging = UNI.yes_no("Force averaging rates for patch simulations?", False)
    force_baseline_averaging = UNI.yes_no("Force averaging rates for baseline?", False)
    
    with NeuralHdf5(default_filename, "a", config=config) as file:
        conn = ConnectivityMatrix(config)
        
        connectivity_group = file.require_group(file.root, connectivity_tag)
        file.require_array(connectivity_group, shift_tag, conn.shift)
        file.require_array(connectivity_group, matrix_tag, conn.EE_connections)
        
        indegree_grp = file.require_group(file.root, indegree_tag)
        indegree, _ = conn.degree(conn.EE_connections)
        file.require_array(indegree_grp, indegree_tag, indegree)
        indegree = indegree.ravel()
        for radius in config.radius:
            for name, center in config.center_range.items():
                center = tuple(int(c) for c in center)
                # Calculate indegree
                patch = DOP.circular_patch(config.rows, tuple(center), float(radius))
                patch_indegree = indegree[patch].mean() * config.synapse.weight
                
                indegree_grp._v_attrs[str(center)] = patch_indegree
        
        activity_group = file.require_group(file.root, activity_tag)
        for baseline_tag in config.baseline_tags:
            if not force_baseline_averaging and baseline_tag in activity_group:
                continue
            logger.info(f"Average baseline rate: {baseline_tag}")
            averaged_baseline_rate = average_rate(baseline_tag, sub_directory=config.sub_dir, config=config)
            
            if averaged_baseline_rate is None:
                logger.warning(f"No simulation data for {baseline_tag} found...")
                continue
            
            # TODO: Alternative would be the seed of the baseline_tag
            file.require_array(activity_group, baseline_tag, averaged_baseline_rate)
                
        for tag in config.get_all_tags():
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
            center  = tuple(int(c) for c in center)
            radius  = UNI.radius_from_tag(tag)
            percent = UNI.split_percentage_from_tag(tag)
            _, seed = UNI.split_seed_from_tag(tag)
            
            center_group = file.require_group(activity_group, str(center))
            radius_group = file.require_group(center_group, str(radius))
            percent_group = file.require_group(radius_group, str(percent))
                        
            if not force_patch_averaging and str(seed) in percent_group:
                continue
            logger.info(f"Average rate: {tag}")
            averaged_rate = average_rate(tag, sub_directory=config.sub_dir, config=config)
            
            if averaged_rate is None:
                logger.warning(f"No simulation data for {tag} found...")
                continue
            
            file.require_array(percent_group, str(seed), averaged_rate)
    

        scanner = dbs.DBScan_Sequences(config)
        
        spikes_group = file.require_group(file.root, spikes_tag)
        for attr in ("spike_threshold", "eps", "min_samples"):
            value = getattr(spikes_group._v_attrs, attr, None)
            if value is not None and config.analysis.sequence[attr] != value:
                force_baseline = True
                force_patch = True
                
        logger.info("Scan Baselines...")
        for tag in config.baseline_tags:
            if not force_baseline and tag in spikes_group:
                continue
            
            spikes, labels = scanner._scan_spike_train(tag, force=force_baseline)
            
            sub_group = file.require_group(spikes_group, tag)
            file.require_array(sub_group, spikes_tag, spikes)
            file.require_array(sub_group, labels_tag, labels)

    
    
        logger.info("Scan Patches...")
        for tag in config.get_all_tags(specific_tag):
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
            center  = tuple(int(c) for c in center)
            radius  = UNI.radius_from_tag(tag)
            percent = UNI.split_percentage_from_tag(tag)
            _, seed = UNI.split_seed_from_tag(tag)
            
            center_group = file.require_group(spikes_group, str(center))
            radius_group = file.require_group(center_group, str(radius))
            percent_group = file.require_group(radius_group, str(percent))
            seed_group = file.require_group(percent_group, "seed" + str(seed))
            
            if not force_patch and spikes_tag in seed_group and labels_tag in seed_group:
                continue
            
            if not PIC.load_rate(tag, sub_directory=config.sub_dir, config=config, dry=True):
                logger.warning(f"No rate found ({tag})...")
                continue
            
            spikes, labels = scanner._scan_spike_train(tag, force=force_patch)
    
            file.require_array(seed_group, spikes_tag, spikes)
            file.require_array(seed_group, labels_tag, labels)
    # all_tags = config.get_all_tags(specific_tag)
    # correlator = SequenceCorrelator(config)
    #
    # logger.info("Count shared sequences")
    # for tag in all_tags:
    #     try:
    #         correlator.count_shared_sequences(tag)
    #     except KeyError:
    #         logger.info(f"{tag}: No detections spots defined.")
    #         continue




#===============================================================================
# METHODS
#===============================================================================









if __name__ == '__main__':
    for base in np.arange(300, 301):
        config.random_locations = {}
        config.landscape.params["base"] = base
        print(base)
        main(config)
    # main()
    UNI.play_beep(5, .1)
