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

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from constants import DATA_DIR, FIGURE_DIR
from lib import yes_no, functimer, prepend_dir

from params import config

#===============================================================================
# CONSTANTS
#===============================================================================
suffix = ".hdf5"

metadata = "metadata"
data_tag = "data"
baseline_tag = "baseline"
seed_tag = "seed"
rate_tag = "rate"


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
class NeuralHdf5(h5.File):

    @property
    def sim_time(self):
        return self["metadata"]["time"].attrs["sim_time"]

    @property
    def dt(self):
        return self["metadata"]["time"].attrs["dt"]

    @property
    def warmup(self):
        return self["metadata"]["time"].attrs["warmup"]


    def __init__(self, *args, config, **kwargs):
        if kwargs.get("file", None):
            file = kwargs.pop("file")
        else:
            args = list(args)
            file = args.pop(0)
        logger.info("Update filename with dir: {file}")
        file = prepend_dir(file, DATA_DIR)
        super().__init__(file, *args, **kwargs)

        if not self.keys():
            self._reset_metadata(config)

        if not self._is_current_metadata(config):
            if yes_no("Overwrite metadata and delete data?"):
                for key in self.keys():
                    logger.info(f"Delete key: {key}")
                    del self[key]
                self._reset_metadata(config)
            else:
                self.flag_save = False
        logger.info("Finishing __init__...")


    def _reset_metadata(self, config):
        logger.info("Reset metadata...")

        metadata_grp = self.create_group(metadata)

        for tag in (time_tag, network_tag):
            tmp_group = metadata_grp.create_group(tag)
            logger.info(f"Create Group: {tag}")
            for attr in general_setting[tag]:
                logger.info(f"Update {tag}: {attr}")
                tmp_group.attrs[attr] = getattr(config, attr)

        keys = (tranfer_function_tag, synapse_tag, drive_tag, landscape_tag)
        for tag in keys:
            tmp_group = metadata_grp.create_group(tag)
            logger.info(f"Create Group: {tag}")
            config_attr = getattr(config, tag)
            for attr in general_setting[tag]:
                logger.info(f"Update {tag}: {attr}")
                tmp_config_attr = getattr(config_attr, attr)
                if isinstance(tmp_config_attr, dict):
                    logger.info(f"Create Subgroup: {attr}")
                    sub_group = tmp_group.create_group(attr)
                    for key, value in tmp_config_attr.items():
                        sub_group.attrs[key] = value
                else:
                    tmp_group.attrs[attr] = tmp_config_attr


    def _is_current_metadata(self, config):
        logger.info("Comparing metadata...")
        metadata_grp = self[metadata]

        # Checks whether all groups are available
        for group in general_setting.keys():
            if group not in metadata_grp.keys():
                logger.warning(f"Group not found: {group}")
                return False

        # Checks those tags that are directly accessable in config
        for tag in [time_tag, network_tag]:
            tmp_group = metadata_grp[tag]
            for attr in general_setting[tag]:
                logger.info(f"Check {tag}: {attr}")
                if not tmp_group.attrs[attr] == getattr(config, attr):
                    logger.warning(f"Unequal attribute ({tag}): {attr}")
                    return False

        # Checks those tags that are not directly accessable in config
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
                            if not sub_group.attrs[key] == value:
                                logger.warning(f"Unequal attribute ({tag}[{attr}]): {key}")
                                return False
                    else:
                        if not tmp_group.attrs[attr] == tmp_config_attr:
                            logger.warning(f"Unequal attribute ({tag}): {attr}")
                            return False
                except KeyError as err:
                    logger.warning(f"{err.__class__.__name__}: '{attr}' not found.")
                    return False
        return True



#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    import lib.pickler as PIC
    import lib.universal as UNI
    filename = f"{config.landscape.mode}_base{config.landscape.params['base']}_lsseed{config.landscape.seed}.hdf5"
    with NeuralHdf5(filename, "a", config=config) as file:
        file.visit(print)
        data_group = file.require_group(data_tag)
        for tag in config.baseline_tags:
            _, seed = UNI.split_seed_from_tag(tag)
            rate = PIC.load_rate(tag, sub_directory=config.sub_dir)

            group_tmp = data_group.require_group(baseline_tag)
            group_tmp = group_tmp.require_group(seed_tag)
            group_tmp = group_tmp.require_group(str(seed))

            rate_tmp = rate.round(decimals=3)
            dataset = group_tmp.require_dataset(rate_tag, shape=rate_tmp.shape, exact=True, dtype=np.float16)
            dataset[()] = rate_tmp

        for tag in config.get_all_tags():
            name, seed = UNI.split_seed_from_tag(tag)
            name = UNI.name_from_tag(tag)
            percentage = UNI.split_percentage_from_tag(tag)
            radius = UNI.radius_from_tag(tag)
            logger.info(f"Update data of {name} (seed: {seed}) {radius}...")





if __name__ == '__main__':
    main()
