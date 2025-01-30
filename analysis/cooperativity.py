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
from lib import universal as UNI
from lib import dopamine as DOP
from analysis.sequence_correlation import SequenceCorrelator
from analysis.lib import DBScan

from plot.animation import Animator
from plot.figconfig import AnimationConfig
from plot.constants import COLOR_MAP_DIFFERENCE

from plot.lib import plot_patch


dbscan_params = {"eps": config.analysis.sequence.eps,
                 "min_samples": config.analysis.sequence.min_samples,}
mask = np.asarray([False, True, True])
detection_spots_tag = "gate-left"

#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    # animate_cooperativity()
    plot_balance()
    # plot_inbalance()



def plot_inbalance():
    tags = config.get_all_tags(detection_spots_tag)

    for tag in tags:
        print(tag)
        # Find shared clusters
        correlator = SequenceCorrelator(config)
        detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(detection_spots_tag)

        df_sequence_at_center = correlator.detect_sequence_at_center(tag, center=detection_spots)
        # Fix the list of these clusters
        time_column_mask = df_sequence_at_center.columns.str.contains('^C\d_time$')
        time_sequence_at_center = df_sequence_at_center.iloc[:, time_column_mask]
        column_mask = df_sequence_at_center.columns.str.contains('^C\d$')
        sequence_at_center = df_sequence_at_center.iloc[:, column_mask]
        # coop_sequences = df_sequence_at_center[sequence_at_center.all(axis=1)]["sequence id"].to_numpy(dtype=int)

        coop_sequences = df_sequence_at_center[(sequence_at_center == mask).all(axis=1)]["sequence id"].to_numpy(dtype=int)
        if not len(coop_sequences):
            logger.info("No sequences found?")
            continue
        print(coop_sequences)
        inv_coop_sequences = df_sequence_at_center[(sequence_at_center == ~mask).all(axis=1)]["sequence id"].to_numpy(dtype=int)


        spikes, labels = PIC.load_spike_train(tag, config=config)
        sequence_counter = 0
        last_sequence = None
        for i, coop_sequence in enumerate(coop_sequences):
            if coop_sequence == last_sequence:
                sequence_counter += 1
            else:
                sequence_counter = 0
                last_sequence = coop_sequence
        # for i, coop_sequence in enumerate(set(coop_sequences[::])):
            idx = np.argwhere(labels == coop_sequence).squeeze() # idx is the id of the coop-sequence

            tmp_spikes = spikes[idx]
            t_min = tmp_spikes[:, 0].min()  # 0 is time column
            t_max = tmp_spikes[:, 0].max()
            # Separate
            # H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5), density=True)
            # Joint
            H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))


            df_index = df_sequence_at_center["sequence id"] == coop_sequence
            colors = np.asarray(["tab:blue", "tab:orange", "tab:green"])
            times_of_crossing = time_sequence_at_center[df_index].iloc[:, mask].values[sequence_counter]
            # Does not distinguish between one or multiple sequences running through the spots
            # times_of_crossing = time_sequence_at_center[df_index].iloc[:, mask].values.flatten()

            # Separate
            plt.figure(f"Seq_{coop_sequence}_{sequence_counter}",
                       figsize=(3., 2.4), tight_layout=True)
            split_low = edges[edges <= times_of_crossing.min()].size
            split_high = edges[edges > times_of_crossing.max()].size
            plt.plot(edges[:split_low], H[:split_low], c="tab:orange", label="B2")
            plt.plot(edges[split_low:-split_high-1], H[split_low:-split_high], c="tab:red", ls="--", label="B2 & M")
            plt.plot(edges[-split_high:], H[-split_high:], c="tab:green", label="M")
            plt.xlim(edges[0], edges[-1])
            # Joint
            # plt.plot(edges[:-1], i+H/H.max())

            # colors = ["tab:blue", "tab:orange", "tab:green"]
            # times_of_crossing = time_sequence_at_center[df_index].iloc[:, mask].values.flatten()
            for t, c in zip(times_of_crossing, colors[mask]):
                # Separate
                plt.figure(f"Seq_{coop_sequence}_{sequence_counter}")
                plt.axvline(t, ls="--", c=c)
                # Joint
                # plt.plot((t, t), (i, i+1), ls="--", c="tab:blue")

            time_margin = 100
            plt.xlim(max(times_of_crossing.min()-time_margin, 0),
                     times_of_crossing.max()+time_margin)



            if not len(inv_coop_sequences):
                continue

            spikes, labels = PIC.load_spike_train(tag, config=config)
            label = {"label": "B1"}
            for j, inv_sequence in enumerate(set(inv_coop_sequences)):
            # for j, inv_sequence in enumerate(set(inv_coop_sequences[::])):
                idx = np.argwhere(labels == inv_sequence).squeeze() # idx is the id of the coop-sequence

                tmp_spikes = spikes[idx]
                t_min = tmp_spikes[:, 0].min()  # 0 is time column
                t_max = tmp_spikes[:, 0].max()
                # H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5), density=True)
                H, edges = np.histogram(tmp_spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))


                df_index = df_sequence_at_center["sequence id"] == inv_sequence
                times_of_inv_crossing = time_sequence_at_center[df_index].iloc[:, ~mask].values.flatten()

                plt.figure(f"Seq_{coop_sequence}_{sequence_counter}")


                for t in times_of_inv_crossing:
                    if t < times_of_crossing.mean():
                        plt.axvline(t, ls="--", c=colors[~mask][0])
                        # Seperate
                        plt.plot(edges[:-1], H, c=colors[~mask][0], zorder=-2, **label)
                        # Joint
                        # plt.plot(edges[:-1], j+H/H.max(), c="k", zorder=-2)
                        label = {}

            fig = plt.figure(f"Seq_{coop_sequence}_{sequence_counter}")
            plt.xlabel("time [au]")
            plt.ylabel("# of activated neurons")
            plt.legend(labelspacing=.05,)
            fig.patch.set_visible(False)  # Hide figure background
            PIC.save_figure(f"Seq_{coop_sequence}_{sequence_counter}", fig, sub_directory=config.sub_dir,
                            facecolor="none", transparent=True)
        plt.show()
        break


def plot_balance():
    tags = config.get_all_tags(detection_spots_tag)
    # Preparation: Get detection spots
    correlator = SequenceCorrelator(config)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(detection_spots_tag)
    neurons_per_detection_spot = np.asarray([DOP.circular_patch(config.rows, center, radius=2) for center in detection_spots])

    for tag in tags:
        logger.info(f"Analyzing {tag}...")

        # Load data and filter by columns (the times and the indicator)
        df_sequence_at_center = correlator.detect_sequence_at_center(tag, center=detection_spots)
        time_column_mask = df_sequence_at_center.columns.str.contains('^C\d_time$')
        time_sequence_at_center = df_sequence_at_center.iloc[:, time_column_mask]
        column_mask = df_sequence_at_center.columns.str.contains('^C\d$')
        sequence_at_center = df_sequence_at_center.iloc[:, column_mask]

        # Find those sequences that cross all detection spots
        coop_sequences = df_sequence_at_center[sequence_at_center.all(axis=1)]["sequence id"].to_numpy(dtype=int)
        coop_sequences_idx = df_sequence_at_center[sequence_at_center.all(axis=1)].index.to_numpy(dtype=int)
        if not len(coop_sequences):
            logger.info("No sequences found?")
            continue
        logger.info(f"Sequence IDs: {coop_sequences}")

        # One level deeper: Load the actual spikes of the cluster
        spikes, labels = PIC.load_spike_train(tag, config=config)
        for i, (coop_sequence, sequence_iter_idx) in enumerate(zip(coop_sequences, coop_sequences_idx)):
            logger.info(f"Sequence ID: {coop_sequence}")
            # idx is the id of the coop-sequence (May contain multiple rounds of STAS)
            idx = np.argwhere(labels == coop_sequence).squeeze()

            db = DBScan(**dbscan_params, n_jobs=-1, algorithm="auto")

            # Run through the spikes until two separate clusters are found (i.e. both pre exist)
            found_two_clusters = False
            t_min = time_sequence_at_center.iloc[sequence_iter_idx, :2].min()
            t_max = time_sequence_at_center.iloc[sequence_iter_idx, 2]
            time_splits = np.linspace(t_min, t_max, 25, dtype=int)
            for t_low, t_high in pairwise(time_splits):
                low  = np.argwhere(spikes[idx][:, 0] >= t_low )[0, 0]
                high = np.argwhere(spikes[idx][:, 0] <= t_high)[-1, 0]
            # for low, high in pairwise(np.linspace(0, spikes[idx].shape[0], 20, dtype=int)):
                logger.info(f"Times: {t_low} to {t_high} (Index: {low}, {high})")
                _, cluster_labels = db.fit_toroidal(spikes[idx][low:high], nrows=config.rows)
                cluster_labels = UNI.squeeze_labels(cluster_labels)
                assert (cluster_labels >= 0).all()
                print(cluster_labels.size)
                if len(set(cluster_labels)) == 2:
                # if not found_two_clusters and len(set(cluster_labels)) == 2:
                    logger.info("Found 2 clusters")
                    found_two_clusters = True
                    continue

                assert (cluster_labels < 2).all()
                assert (cluster_labels >= 0).all()
                if found_two_clusters and len(set(cluster_labels)) == 1:
                    # Take the lower boundary of the merged cluster
                    # to cluster the both pre-sequences
                    __counter = 0
                    while __counter <= 10 and len(set(cluster_labels)) == 1:
                        low  = np.argwhere(spikes[idx][:, 0] >= (t_low-3*__counter))[0, 0]
                        cluster_spikes, cluster_labels = db.fit_toroidal(spikes[idx][:low], nrows=config.rows)
                        cluster_labels = UNI.squeeze_labels(cluster_labels)
                        assert (cluster_labels >= 0).all()

                        __counter += 1
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
                    logger.info(f"Set of cluster labels: {set(cluster_labels)}")
                    assert len(set(cluster_labels)) == 2

                    colors = ["tab:blue", "tab:orange", "tab:green"]
                    # Analyze the individual clusters
                    for i in range(len(set(cluster_labels))):
                        # Cluster the number of active neurons per time step
                        tmp_spikes = cluster_spikes[cluster_labels == i]
                        # Check to which branch the cluster belongs
                        # Assumption: At least one spike will be detected at the center of the detection spot
                        # tmp_spikes[:, 1:] == # of all coordinates, is any in detection spot
                        time_buffer = 20
                        for d, detection_spot in enumerate(detection_spots):
                            time_crossing = time_sequence_at_center.iloc[sequence_iter_idx, d]
                            time_idx = np.logical_and(tmp_spikes[:, 0] > (time_crossing-time_buffer), tmp_spikes[:, 0] < (time_crossing+time_crossing))
                            crossed_spot = (tmp_spikes[time_idx, 1:][:, np.newaxis] == config.coordinates[neurons_per_detection_spot[d]]).all(axis=-1).any()
                            if crossed_spot:
                                break
                            else:
                                d = None
                        # hist by time
                        H, edges = hist_spike_over_time(tmp_spikes)

                        plt.figure(f"{tag}_coop_{coop_sequence}_{sequence_iter_idx}",
                                   figsize=(3, 2.4), tight_layout=True)
                        plt.plot(edges[1:], H, label=f"B{d+1}", c=colors[d])

                    # Cluster and plot the joint cluster
                    cluster_spikes, cluster_labels = db.fit_toroidal(spikes[idx][low:], nrows=config.rows)
                    cluster_labels = UNI.squeeze_labels(cluster_labels)
                    H, edges = hist_spike_over_time(cluster_spikes)
                    plt.plot(edges[:-1], H, c="tab:green", label="M")
                    # Plot up to k=50 time steps in a scaled version
                    plt.plot(edges[:-1][:50], H[:50]/2, c="tab:green", ls="--", label="scaled")

                    colors = ["tab:blue", "tab:orange", "tab:green"]
                    #  Set the indication for when the STAS crosses the detection spot
                    for t, c in zip(time_sequence_at_center.iloc[sequence_iter_idx].values.flatten(), colors):
                        plt.figure(f"{tag}_coop_{coop_sequence}_{sequence_iter_idx}")
                        plt.axvline(t, ls="--", c=c)

                    # Set the limits
                    times = time_sequence_at_center.iloc[sequence_iter_idx].values
                    time_margin = 50
                    plt.xlim(max(times[:2].min()-time_margin, 0),
                                 times[2]+time_margin)
                    plt.legend(labelspacing=.05,)

                    fig = plt.figure(f"{tag}_coop_{coop_sequence}_{sequence_iter_idx}")
                    plt.xlabel("time [au]")
                    plt.ylabel("# of activated neurons")

                    fig.patch.set_visible(False)  # Hide figure background
                    PIC.save_figure(f"{tag}_coop", fig, sub_directory=config.sub_dir,
                                    facecolor="none", transparent=True)
                    # plt.show()
                    logger.info("Break the splitting in time...")
                    break
        # logger.info("Hit the break for tag in tags...")
        # break
    plt.show()



def hist_spike_over_time(spikes:np.ndarray):
    t_min = spikes[:, 0].min()  # 0 is time column
    t_max = spikes[:, 0].max()
    H, edges = np.histogram(spikes[:, 0], bins=np.arange(t_min-0.5, t_max+0.5))
    return H, edges






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
        if mask.all() and not len(coop_sequences):
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

        if mask.all() and not len(starts):
            logger.info("No shared sequence across all detection spots found.")
            continue

        coop_index = 0
        # anim_kwargs = {"start": int(max(starts[coop_index] - 0, 0)), "step": 1, "stop": int(stops[coop_index]), "interval": 100}
        anim_kwargs = {"start": 3000, "step": 1, "stop": 3200, "interval": 100}

        # animator = Animator(config, AnimationConfig)
        # animator.animate([bs_tag], **anim_kwargs, add_spikes=False)
        # print(100*"*")

        # syn_inputs = PIC.load_synaptic_input(bs_tag, sub_directory=config.sub_dir)
        # syn_inputs = syn_inputs[:config.rows**2, :config.rows**2]
        # animator.baseline_figure("synaptic"+bs_tag, syn_inputs,
        #                          norm=(-200, 200), cmap=COLOR_MAP_DIFFERENCE,
        #                          **anim_kwargs)
        print(100*"-")

        animator = Animator(config, AnimationConfig)
        animator.animate([tag], **anim_kwargs, add_spikes=False)
        for ds in detection_spots:
            plot_patch(ds, radius=2, width=config.rows)

        syn_inputs = PIC.load_synaptic_input(tag, sub_directory=config.sub_dir)
        syn_inputs = syn_inputs[:config.rows**2, :config.rows**2]
        animator.baseline_figure("synaptic"+tag, syn_inputs,
                                 norm=(-200, 200), cmap=COLOR_MAP_DIFFERENCE,
                                 **anim_kwargs)
        for ds in detection_spots:
            plot_patch(ds, radius=2, width=config.rows)
        animator.show()
        break

    animator.show()

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
