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
config = GateConfig()

from lib import pickler as PIC
from analysis.sequence_correlation import SequenceCorrelator
from analysis.lib import DBScan

from plot.animation import Animator
from plot.figconfig import AnimationConfig
from plot.constants import COLOR_MAP_DIFFERENCE



dbscan_params = {"eps": config.analysis.sequence.eps,
                 "min_samples": config.analysis.sequence.min_samples,}

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    # animate_cooperativity()
    # plot_balance()
    plot_inbalance()



def plot_inbalance():
    detection_spots_tag = "gate-left"
    tags = config.get_all_tags(detection_spots_tag)

    for tag in tags:
        print(tag)
        # Find shared clusters
        correlator = SequenceCorrelator(config)
        detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(detection_spots_tag)

        df_sequence_at_center = correlator.detect_sequence_at_center(tag, center=detection_spots)
        # Fix the list of these clusters
        sequence_at_center = df_sequence_at_center.iloc[:, 1:]
        # coop_sequences = df_sequence_at_center[sequence_at_center.all(axis=1)]["sequence id"].to_numpy(dtype=int)
        mask = np.asarray([False, True, True])
        coop_sequences = df_sequence_at_center[(sequence_at_center == mask).all(axis=1)]["sequence id"].to_numpy(dtype=int)
        if not len(coop_sequences):
            logger.info("No sequences found?")
            continue
        print(coop_sequences)

        spikes, labels = PIC.load_spike_train(tag, config=config)
        for i, coop_sequence in enumerate(set(coop_sequences[::])):
            idx = np.argwhere(labels == coop_sequence).squeeze() # idx is the id of the coop-sequence

            tmp_spikes = spikes[idx]
            t_min = tmp_spikes[:, 0].min()  # 0 is time column
            t_max = tmp_spikes[:, 0].max()
            H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5), density=True)
            plt.plot(edges[:-1], i+H/H.max())

        coop_sequences = df_sequence_at_center[(sequence_at_center == ~mask).all(axis=1)]["sequence id"].to_numpy(dtype=int)
        if not len(coop_sequences):
            continue

        spikes, labels = PIC.load_spike_train(tag, config=config)
        for i, coop_sequence in enumerate(set(coop_sequences[::])):
            idx = np.argwhere(labels == coop_sequence).squeeze() # idx is the id of the coop-sequence

            tmp_spikes = spikes[idx]
            t_min = tmp_spikes[:, 0].min()  # 0 is time column
            t_max = tmp_spikes[:, 0].max()
            H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5), density=True)
            plt.plot(edges[:-1], i+H/H.max(), c="k")
        plt.show()
        break


def plot_balance():
    tag = config.baseline_tags[0]
    detection_spots_tag = "gate-left"
    tags = config.get_all_tags(detection_spots_tag)
    tag = tags[3]

    for tag in tags:
        print(tag)
        # Find shared clusters
        correlator = SequenceCorrelator(config)
        detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(detection_spots_tag)

        df_sequence_at_center = correlator.detect_sequence_at_center(tag, center=detection_spots)
        # Fix the list of these clusters
        sequence_at_center = df_sequence_at_center.iloc[:, 1:]
        coop_sequences = df_sequence_at_center[sequence_at_center.all(axis=1)]["sequence id"].to_numpy(dtype=int)


        if not len(coop_sequences):
            logger.info("No sequences found?")
            continue
        print(coop_sequences)

        spikes, labels = PIC.load_spike_train(tag, config=config)
        for i, coop_sequence in enumerate(coop_sequences[::-1]):
            idx = np.argwhere(labels == coop_sequence).squeeze() # idx is the id of the coop-sequence

            db = DBScan(**dbscan_params, n_jobs=-1, algorithm="auto")

            found_two_clusters = False
            for low, high in pairwise(np.linspace(0, spikes[idx].shape[0], 20, dtype=int)):
                print(low, high)
                _, cluster_labels = db.fit_toroidal(spikes[idx][low:high], nrows=config.rows)
                assert (cluster_labels >= 0).all()
                print(cluster_labels.size)
                if len(set(cluster_labels)) == 2:
                # if not found_two_clusters and len(set(cluster_labels)) == 2:
                    print("Found 2 clusters")
                    found_two_clusters = True
                    # if tag == "gate-left_6_50_10_3":
                    #     H, _, _ = np.histogram2d(*spikes[idx][low:high][:, 1:].T, bins=np.arange(-0.5, config.rows+1))
                    #     plt.figure()
                    #     plt.imshow(H.T, origin="lower")
                    #     plt.show()
                    continue

                assert (cluster_labels < 2).all()
                assert (cluster_labels >= 0).all()
                if found_two_clusters and len(set(cluster_labels)) == 1:
                    # Take the lower boundary of the merged cluster
                    # to cluster the both pre-sequences
                    cluster_spikes, cluster_labels = db.fit_toroidal(spikes[idx][:low], nrows=config.rows)
                    assert (cluster_labels >= 0).all()
                    # if len(set(cluster_labels)) != 2:
                    #     H, _, _ = np.histogram2d(*spikes[idx][:high][:, 1:].T, bins=np.arange(-0.5, config.rows+1))
                    #     plt.figure("to high")
                    #     plt.imshow(H.T, origin="lower")
                    #     # plt.show()
                    #     plt.figure("to low")
                    #     H, _, _ = np.histogram2d(*spikes[idx][:low][:, 1:].T, bins=np.arange(-0.5, config.rows+1))
                    #     plt.imshow(H.T, origin="lower")
                    #     plt.show()
                    # It may happen that the DBSCAN still only finds one cluster.
                    # That may arise to a (or a few) spikes that are close in space but sliced away in the low:high process.
                    # Consequently -> continue?
                    assert len(set(cluster_labels)) == 2
                    for i in range(2):
                        tmp_spikes = cluster_spikes[cluster_labels == i]
                        t_min = tmp_spikes[:, 0].min()  # 0 is time column
                        t_max = tmp_spikes[:, 0].max()
                        H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))

                        plt.figure(f"{tag}_coop")
                        plt.plot(edges[:-1], H)
                    cluster_spikes, cluster_labels = db.fit_toroidal(spikes[idx][low:], nrows=config.rows)
                    tmp_spikes = cluster_spikes
                    t_min = tmp_spikes[:, 0].min()  # 0 is time column
                    t_max = tmp_spikes[:, 0].max()
                    H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))
                    plt.plot(edges[:-1], H/2)
                        # plt.hist(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))
                    plt.show()
                    break
                break






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
        idx = np.argwhere(labels == sequence).squeeze()
        start = spikes[idx][:, 0].min() # 0 for only time
        starts[i] = start
        stop = spikes[idx][:, 0].max() # 0 for only time
        stops[i] = stop
    return starts, stops








if __name__ == '__main__':
    main()
