#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:

Description:

TODO:
    - Test with other landscapes that have no e.g. params.
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

import tables as tb
import numpy as np
import matplotlib.pyplot as plt
import datetime

from constants import DATA_DIR, FIGURE_DIR
from lib import yes_no, functimer, prepend_dir
import lib.universal as UNI

from params import config

#===============================================================================
# CONSTANTS
#===============================================================================
suffix = ".hdf5"
default_filename = "data.hdf5"

metadata = "metadata"
data_tag = "data"
baseline_tag = "baseline"
patch_tag = "patch"
seed_tag = "seed"
rate_tag = "rate"

connectivity_tag = "connectivity"
shift_tag = "shift"
matrix_tag = "W"
indegree_tag = "indegree"

activity_tag = "activity"
avg_rate_tag = "avg_rate"
spikes_tag   = "pseudospikes"
labels_tag   = "labels"


# The metadata structure is defined here. All values of the items
# in the list are compared.
time_tag = "time"
network_tag = "network"
tranfer_function_tag = "transfer_function"
synapse_tag = "synapse"
drive_tag = "drive"
landscape_tag = "landscape"
general_setting = {
    time_tag: ("warmup", "sim_time", "defaultclock_dt"),
    network_tag: ("rows", "tau"),
    tranfer_function_tag: ("offset", "slope"),
    synapse_tag: ("weight", "EI_factor"),
    drive_tag: ("mean", "std"),
    landscape_tag: ("mode", "stdE", "stdI", "shift", "connection_probability", "params"),
}

#===============================================================================
# CLASS
#===============================================================================
class NeuralHdf5(tb.File):

    @property
    def sim_time(self):
        return self[metadata_tag][time_tag].attrs["sim_time"]

    @property
    def dt(self):
        return self[metadata_tag][time_tag].attrs["dt"]

    @property
    def warmup(self):
        return self[metadata_tag][time_tag].attrs["warmup"]
    
    @property
    def shift(self):
        return self.root[connectivity_tag][shift_tag].read()
    
    
    def get_indegree(self, center:tuple=None):
        group = self.root[indegree_tag]
        if center == None:
            return group[indegree_tag].read()
        
        center  = tuple(int(c) for c in center)
        return getattr(group._v_attrs, str(center))
            


    def __init__(self, *args, config, **kwargs):
        if kwargs.get("file", None):
            file = kwargs.pop("file")
        else:
            args = list(args)
            file = args.pop(0)
        logger.info(f"Update filename with dir: {file}")
        file = prepend_dir(file, config.sub_dir)
        file = prepend_dir(file, DATA_DIR)
        logger.info(f"File: {file}")
        super().__init__(file, *args, **kwargs)

        if metadata not in self.root:
            logger.info("Resetting metadata...")
            self._reset_metadata(config)

        if not self._is_current_metadata(config):
            if yes_no("Different metadata: Rename (y) or abort (n)?"):
                prefix = datetime.datetime.now().strftime(dateformat)
                path = prepend_dir(prefix + "_" + file, DATA_DIR)
                self.copy_file(path)
                for child in self.root._v_children.values():
                    self.remove_node(child, recursive=True)
                self._reset_metadata(metadata)
            else:
                raise FileExistsError
        logger.info("Finishing __init__...")


    def _reset_metadata(self, config):
        logger.info("Reset metadata...")

        metadata_grp = self.require_group(self.root, metadata)

        for tag in (time_tag, network_tag):
            tmp_group = self.require_group(metadata_grp, tag)
            logger.info(f"Create Group: {tag}")
            for attr in general_setting[tag]:
                logger.info(f"Update {tag}: {attr}")
                tmp_group._v_attrs[attr] = getattr(config, attr)

        keys = (tranfer_function_tag, synapse_tag, drive_tag, landscape_tag)
        for tag in keys:
            tmp_group = self.require_group(metadata_grp, tag)
            logger.info(f"Create Group: {tag}")
            config_attr = getattr(config, tag)
            for attr in general_setting[tag]:
                logger.info(f"Update {tag}: {attr}")
                tmp_config_attr = getattr(config_attr, attr)
                if isinstance(tmp_config_attr, dict):
                    logger.info(f"Create Subgroup: {attr}")
                    sub_group = self.require_group(tmp_group, attr)   
                    for key, value in tmp_config_attr.items():
                        sub_group._v_attrs[key] = value
                else:
                    tmp_group._v_attrs[attr] = tmp_config_attr


    def _is_current_metadata(self, config):
        logger.info("Comparing metadata...")
        metadata_grp = self.root[metadata]

        # Checks whether all groups are available
        for group in general_setting.keys():
            if group not in metadata_grp:
                logger.warning(f"Group not found: {group}")
                return False

        # Checks those tags that are directly accessable in config
        for tag in [time_tag, network_tag]:
            tmp_group = metadata_grp[tag]
            for attr in general_setting[tag]:
                logger.info(f"Check {tag}: {attr}")
                if not tmp_group._v_attrs[attr] == getattr(config, attr):
                    logger.warning(f"Unequal attribute ({tag}): {attr}")
                    return False

        # Checks those tags that are not directly accessible in config
        keys = (tranfer_function_tag, synapse_tag, drive_tag, landscape_tag)
        for tag in keys:
            config_attr = getattr(config, tag)
            tmp_group = metadata_grp[tag]
            for attr in general_setting[tag]:
                logger.info(f"Check {tag}: {attr}")

                tmp_config_attr = getattr(config_attr, attr)
                try:
                    if isinstance(tmp_config_attr, dict):
                        sub_group = tmp_group[attr]
                        for key, value in tmp_config_attr.items():
                            if not sub_group._v_attrs[key] == value:
                                logger.warning(f"Unequal attribute ({tag}[{attr}]): {key}")
                                return False
                    else:
                        if not tmp_group._v_attrs[attr] == tmp_config_attr:
                            logger.warning(f"Unequal attribute ({tag}): {attr}")
                            return False
                except KeyError as err:
                    logger.warning(f"{err.__class__.__name__}: '{attr}' not found.")
                    return False
        return True
    
    
    def require_group(self, where:str, name:str, *args, **kwargs)->tb.Group:
        """Extension of the method {create_group} with same signature."""
        try:
            return self.get_node(where, name)
        except tb.NoSuchNodeError:
            logger.info(f"Create new Group: {name}")
            return self.create_group(where, name, *args, **kwargs)
        
        
    def require_table(self, where:str, name:str, *args, **kwargs)->tb.Table:
        """Extension of the method {create_table} with same signature."""
        try:
            return self.get_node(where, name)
        except tb.NoSuchNodeError:
            logger.info(f"Create new Table: {name}")
            return self.create_table(where, name, *args, **kwargs)


    def require_array(self, where:str, name:str, obj:np.ndarray, force:bool=False, *args, **kwargs):
        try:
            node = self.get_node(where, name)
        except tb.NoSuchNodeError:
            node = None
            
        if node is None:
            return self.create_array(where, name, obj, *args, **kwargs)
        
        
        if force:
            self.remove_node(where, name)
            node = self.create_array(where, name, obj, *args, **kwargs)
        if obj.dtype != node.dtype:
            logger.info(f"Different types -> Save with new type {obj.dtype}...")
            self.remove_node(where, name)
            node = self.create_array(where, name, obj, *args, **kwargs)
        return node
    
        
    def reset_sequence_duration_and_count(self, tag:str, is_baseline:bool = False):
        if is_baseline:
            group = self.root.pseudospikes[tag]
        else:
            name    = UNI.name_from_tag(tag)
            center  = config.center_range[name]
            center  = tuple(int(c) for c in center)
            radius  = UNI.radius_from_tag(tag)
            percent = UNI.split_percentage_from_tag(tag)
            _, seed = UNI.split_seed_from_tag(tag)
            group = self.root.pseudospikes[str(center)][str(radius)][str(percent)][str(seed)]
            
        if hasattr(group[spikes_tag]._v_attrs, "durations"):
            del group[spikes_tag]._v_attrs["durations"]
        if hasattr(group[labels_tag]._v_attrs, "count"):
            del group[labels_tag]._v_attrs["count"]


    # TODO: is tag the correct variable name?
    @functimer
    def get_sequence_duration_and_count(self, tag:str, is_baseline:bool = False):
        if is_baseline:
            group = self.root.pseudospikes[tag]
        else:
            name = UNI.name_from_tag(tag)
            center = config.center_range[name]
            center = tuple(int(c) for c in center)
            radius = UNI.radius_from_tag(tag)
            percent= UNI.split_percentage_from_tag(tag)
            _, seed = UNI.split_seed_from_tag(tag)
            group = self.root.pseudospikes[str(center)][str(radius)][str(percent)]["seed" + str(seed)]
         
        durations = getattr(group[spikes_tag]._v_attrs, "durations", None)
        count = getattr(group[labels_tag]._v_attrs, "count", None)
        if durations is not None and count is not None:
            return durations, count
        
        spikes = group[spikes_tag].read()
        labels = group[labels_tag].read()
        from plot.sequences import _get_durations
        durations = _get_durations(spikes[:, 0], labels)
        group[spikes_tag]._v_attrs["durations"] = durations
        group[labels_tag]._v_attrs["count"] = labels.max()
        return durations, labels.max()
    
    
    @functimer
    def get_spikes_with_labels(self, tag:str, is_baseline:bool = False):
        if is_baseline:
            group = self.root.pseudospikes[tag]
        else:
            name = UNI.name_from_tag(tag)
            center = config.center_range[name]
            center = tuple(int(c) for c in center)
            radius = UNI.radius_from_tag(tag)
            percent= UNI.split_percentage_from_tag(tag)
            _, seed = UNI.split_seed_from_tag(tag)
            group = self.root.pseudospikes[str(center)][str(radius)][str(percent)]["seed" + str(seed)]
        spikes = group[spikes_tag].read()       
        labels = group[labels_tag].read()
        return spikes, labels
     
    
    def get_average_rate(self, tag:str, is_baseline:bool = False):
        if is_baseline:
            group = self.root[activity_tag][tag]
        else:
            name = UNI.name_from_tag(tag)
            center = config.center_range[name]
            center = tuple(int(c) for c in center)
            radius = UNI.radius_from_tag(tag)
            percent= UNI.split_percentage_from_tag(tag)
            _, seed = UNI.split_seed_from_tag(tag)
            group = self.root[activity_tag][str(center)][str(radius)][str(percent)]["seed" + str(seed)]
            
        return group.read()
        
            
#===============================================================================
# MAIN METHOD
#===============================================================================

def main():
    import lib.pickler as PIC
    import lib.universal as UNI
    import lib.dopamine as DOP
    from lib.connectivitymatrix import ConnectivityMatrix
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
                print(name, center)
                # Calculate indegree
                
                patch = DOP.circular_patch(config.rows, tuple(center), float(radius))
                patch_indegree = indegree[patch].mean() * config.synapse.weight
                
                indegree_grp._v_attrs[str(center)] = patch_indegree
        file.flush()
        
        
        
        
        # data_group = file.require_group(data_tag)
        # for tag in config.baseline_tags:
        #     _, seed = UNI.split_seed_from_tag(tag)
        #     rate = PIC.load_rate(tag, sub_directory=config.sub_dir)
        #
        #     group_tmp = data_group.require_group(baseline_tag)
        #     group_tmp = group_tmp.require_group(seed_tag)
        #     group_tmp = group_tmp.require_group(str(seed))
        #
        #     rate_tmp = rate.round(decimals=3)
        #     dataset = group_tmp.require_dataset(rate_tag, shape=rate_tmp.shape, exact=True, dtype=np.float16)
        #     dataset[()] = rate_tmp
        #
        # for tag in config.get_all_tags():
        #     name, seed = UNI.split_seed_from_tag(tag)
        #     name = UNI.name_from_tag(tag)
        #     percentage = UNI.split_percentage_from_tag(tag)
        #     radius = UNI.radius_from_tag(tag)
        #     logger.info(f"Update data of {name} (seed: {seed}) {radius}...")





if __name__ == '__main__':
    main()
