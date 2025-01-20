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
# from itertools
def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b


from params.config_handler import config
from params.motifconfig import GateConfig, CoopConfig, Gate2Config, Gate3Config
allowed_configs = (GateConfig, CoopConfig, Gate2Config, Gate3Config)
if type(config) not in allowed_configs:
    print("No valid config given. Fall back to default.")
    config = GateConfig()
config.landscape.seed = 1

from lib import pickler as PIC
from analysis.sequence_correlation import SequenceCorrelator
from analysis.lib import DBScan

from plot.animation import Animator
from plot.figconfig import AnimationConfig
from plot.constants import COLOR_MAP_DIFFERENCE


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    # animate_cooperativity()
    plot_balance()



def plot_balance():
    dbscan_params = {"eps": config.analysis.sequence.eps,
                     "min_samples": config.analysis.sequence.min_samples,}
    tag = config.baseline_tags[0]
    detection_spots_tag = "gate-left"
    # Find shared clusters
    correlator = SequenceCorrelator(config)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(detection_spots_tag)

    sequence_at_center = correlator.detect_sequence_at_center(tag, center=detection_spots)
    # Fix the list of these clusters
    coop_sequences = np.argwhere(sequence_at_center.all(axis=1)).ravel()
    if not len(coop_sequences):
        logger.info("No sequences found?")

    spikes, labels = PIC.load_spike_train(tag, config=config)
    for i, coop_sequence in enumerate(coop_sequences[2:]):
        idx = np.argwhere(labels == coop_sequence).squeeze() # idx is the id of the coop-sequence


        start = spikes[idx][:, 0].min()
        stop = spikes[idx][:, 0].max()
        print(start, stop)

        splits = np.linspace(start, stop, 2, dtype=int)
        both_sequences_exist = False
        for a, b in pairwise(splits):
            # split_spikes = np.logical_and(spikes[idx][:, 0] >= a, spikes[idx][:, 0] < b)
            split_spikes = spikes[idx][:, 0] < b

            db = DBScan(**dbscan_params, n_jobs=-1, algorithm="auto")
            data, labels = db.fit_toroidal(spikes[idx][split_spikes], nrows=config.rows)
            assert (labels >= 0).all()



            if np.unique(labels).size > 1:
                both_sequences_exist = True

            if both_sequences_exist and np.unique(labels).size == 1:
                print("A", a, "B", b)
                break

        split_spikes = spikes[idx][:, 0] < a

        # coordinates_bs = spikes[idx][split_spikes][:, 1:]
        coordinates_bs = spikes[idx][:, 1:]
        H_bs, _, _ = np.histogram2d(*coordinates_bs.T, bins=np.arange(-0.5, config.rows))

        plt.figure(tag)
        plt.title(f"Sequences")
        cmap = "hot"
        from plot.lib import create_image
        im = create_image(H_bs.T, cmap=cmap)
        plt.colorbar(im)
        plt.show()


        # db = DBScan(**dbscan_params, n_jobs=-1, algorithm="auto")
        # data, labels = db.fit_toroidal(spikes[idx][split_spikes], nrows=config.rows)
        # assert (labels >= 0).all()
        # assert labels.size == 2

        # plt.figure()
        # split_spikes[labels == 0]
        break



    # spikes, labels = PIC.load_spike_train(tag, config=config)
    # starts, stops = get_starts_and_stops(spikes, labels, coop_sequences)

    # logger.info(f"Starts: {starts}")
    # if not len(starts):
    #     logger.info("No shared sequence across all detection spots found.")


    # Animate baseline




def animate_cooperativity():
    # Load shared clusters, i.e. to see which cluster acutally is shared across 'all'
    specific_tag = "gate-left"
    tags = config.get_all_tags(specific_tag)
    for tag in tags:
        logger.info(f"Start for tag: {tag}")
        correlator = SequenceCorrelator(config)
        detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag)
        bs_tag = config.get_baseline_tag_from_tag(tag)
        sequence_at_center = correlator.detect_sequence_at_center(tag, center=detection_spots)
        # Fix the list of these clusters
        coop_sequences = np.argwhere(sequence_at_center.all(axis=1)).ravel()
        if not len(coop_sequences):
            logger.info("No sequences found?")
            continue


        # Determine start of each of coop sequences
        starts = np.zeros(coop_sequences.size)
        stops = np.zeros(coop_sequences.size)
        spikes, labels = PIC.load_spike_train(tag, config=config)
        for i, coop_sequence in enumerate(coop_sequences):
            idx = np.argwhere(labels == coop_sequence).squeeze()
            start = spikes[idx][:, 0].min() # 0 for only time
            starts[i] = start
            stop = spikes[idx][:, 0].max() # 0 for only time
            stops[i] = stop
        print("Starts: ", starts)

        if not len(starts):
            logger.info("No shared sequence across all detection spots found.")
            continue

        coop_index = 0
        anim_kwargs = {"start": int(max(starts[coop_index] - 0, 0)), "step": 1, "stop": int(stops[coop_index]), "interval": 100}

        animator = Animator(config, AnimationConfig)
        animator.animate([bs_tag], **anim_kwargs, add_spikes=False)

        syn_inputs = PIC.load_synaptic_input(bs_tag, sub_directory=config.sub_dir)
        syn_inputs = syn_inputs[:config.rows**2, :config.rows**2]
        animator.baseline_figure("synaptic"+bs_tag, syn_inputs,
                                 norm=(-200, 200), cmap=COLOR_MAP_DIFFERENCE,
                                 **anim_kwargs)

        animator = Animator(config, AnimationConfig)
        animator.animate([tag], **anim_kwargs, add_spikes=False)

        syn_inputs = PIC.load_synaptic_input(tag, sub_directory=config.sub_dir)
        syn_inputs = syn_inputs[:config.rows**2, :config.rows**2]
        animator.baseline_figure("synaptic"+tag, syn_inputs,
                                 norm=(-200, 200), cmap=COLOR_MAP_DIFFERENCE,
                                 **anim_kwargs)
        break

    plt.show()

#===============================================================================
# METHODS
#===============================================================================

def get_starts_and_stops(spikes, labels, sequences):
    # Determine start of each of coop sequences
    starts = np.zeros(sequences.size)
    stops = np.zeros(sequences.size)

    for i, sequence in enumerate(sequences):
        idx = np.argwhere(labels == coop_sequence).squeeze()
        start = spikes[idx][:, 0].min() # 0 for only time
        starts[i] = start
        stop = spikes[idx][:, 0].max() # 0 for only time
        stops[i] = stop
    return starts, stops








if __name__ == '__main__':
    main()
