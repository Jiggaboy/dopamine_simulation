#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:20:21 2022

@author: hauke
"""

import matplotlib.pyplot as plt
import numpy as np

from plot.lib import SequenceCounter

import lib.pickler as PIC
import universal as UNI


from figure_generator.connectivity_distribution import set_layout, SPINE_WIDTH

def main():
    from params import PerlinConfig

    cf = PerlinConfig()

    all_tags = cf.get_all_tags()
    #all_tags = [t for t in all_tags if t.startswith("repeater")]

    def scatter(x, data, **kwargs):
        plot_to_scatter = {"ls": "None", "marker": "o"}
        try:
            line, = plt.plot(np.full(shape=len(data), fill_value=x), data, **plot_to_scatter, **kwargs)
        except TypeError:
            line, = plt.plot(x, data, **plot_to_scatter, **kwargs)
        return line


    def scatter_baseline_patch(x, sequence, center_idx:int, distance:float=1., **kwargs):
        MS = 30
        scatter(x, sequence.baseline[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)
        scatter(x+distance, sequence.patch[center_idx], markerfacecolor="white", markersize=MS / 2, **kwargs)

        scatter(x, sequence.baseline_avg[center_idx], markersize=MS, **kwargs)
        return scatter(x+distance, sequence.patch_avg[center_idx], markersize=MS, **kwargs)



    KTH_GREEN = 176, 201, 43
    KTH_PINK = 216, 84, 151
    KTH_GREY = 101, 101, 108
    colors = "green", "blue", "orange"
    colors = KTH_GREEN, KTH_PINK, KTH_GREY
    for tag in all_tags:
        sequence = PIC.load_sequence(tag, sub_directory=cf.sub_dir)
        print(tag)
        plt.figure(tag, figsize=(6, 6))
        plt.xlim(-.1, 1.5)
        plt.ylim(0, 250)
        plt.yticks([0, 100, 200])
        handles = []
        for idx, (center, c) in enumerate(zip(sequence.center, colors)):
            c = np.asarray(c) / 255
            handle = scatter_baseline_patch(idx * .2, sequence, idx, c=c)
            handles.append(handle)
            print(f"{center}: {sequence.baseline_avg[idx]} -> {sequence.patch_avg[idx]}")
        bold_spines(plt.gca())
        plt.ylabel("# Sequences")
        plt.xticks([.1, 1.1], labels=["w/o patch", "w/ patch"])
        plt.tight_layout()
        plt.savefig(UNI.get_fig_filename(tag + "_sequences", format_="svg"), format="svg")
        plt.legend(handles=handles, labels=sequence.center)
    plt.show()


def bold_spines(ax, width:float=SPINE_WIDTH):
    # ax.set_xticks([])
    tick_params = {"width": SPINE_WIDTH, "length": SPINE_WIDTH * 3, "labelleft": True, "labelbottom": True}
    ax.tick_params(**tick_params)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
            ax.spines[s].set_linewidth(SPINE_WIDTH)

def plot_passing_sequences_pre_post(patches:np.ndarray, postfix:str, figname:str, title:str=None, details:tuple=None, details_in_title:bool=True):
    # from analysis.passing_sequences import number_of_sequences
    # Resetting the color cycle, why?
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    width = 2.
    weigth = 5
    plt.figure(figname, figsize=(4, 2.8))

    counts = [number_of_sequences(p.nonzero()[0], avg=False, postfix=postfix) for p in patches]

    heights, bins, handlers_neurons = plt.hist(counts, bins=np.arange(0, 250, 10), color=[colors[0], colors[2]])
    handlers_avg = []
    for idx, p in enumerate(patches):
        c_idx = 2 * idx + 1
        count = number_of_sequences(p, avg=True, postfix=postfix)
        print(f"{postfix}: {count} - {idx}")
        _, _, handler = plt.hist(count, color=colors[c_idx], bins=[count-width, count+width], weights=[weigth])
        handlers_avg.append(handler)
    plt.title(title)
    handlers = (
        handlers_neurons[0],
        handlers_avg[0],
        handlers_neurons[1],
        handlers_avg[1],
        )
    # handlers = handlers_neurons
    labels = (
        "Main: Ind. neurons",
        "Main: Avg. activity",
        "Low. pathway: Ind. neurons",
        "Low. pathway: Avg. activity",
        )
    # repeater
    # plt.xlim(0, 255)
    # plt.ylim(0, 11)
    # linker
    # plt.xlim(0, 210)
    # plt.ylim(0, 12)
    # activator
    plt.xlim(0, 250)
    plt.ylim(0, 11)
    plt.legend(handlers, labels)


if __name__ == "__main__":
    main()
