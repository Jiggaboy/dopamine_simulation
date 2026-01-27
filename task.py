#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary: 
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
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from class_lib import ExternalDrive
import lib.pickler as PIC
from lib.connectivitymatrix import ConnectivityMatrix
from params import MotifConfig
from lib.brian import BrianSimulator
import lib.dopamine as DOP
import lib.universal as UNI

from itertools import product
from analysis.sequence_correlation import SequenceCorrelator

#===============================================================================
# TaskConfig
#===============================================================================
class TaskConfig(MotifConfig):
    PERCENTAGES = .1, -.1
    
    radius = 5
    AMOUNT_NEURONS = 40,
    
    drive = ExternalDrive(5., 30., seeds=(0, ))
    
    sim_time = 2000
    center_range = OrderedDict({
        # everything should be offset by 1, center is 49
        "start-left":  (39, 82),
        "start-right": (59, 82),
        "gate-left":   (55, 70),
        "gate-right":  (43, 70),
        "select-left":   (54, 18),
        "select-right":  (44, 18),
        "repeat":  (49, 40),      
    })
    
    task = {
        "task-A": (
            ("start-left", .2), 
            ("gate-left", -.1), 
            ("repeat", .1), 
            ("select-left", .2), 
        ), 
    }
    
    
    def _add_detection_spots(self) -> list:
        detection_spots = []
        # start-left, before repeat, before select, select-left, select-right
        center = ((40, 76), (49, 51), (49, 28), (37, 10), (61, 10)) 
        UNI.append_spot(detection_spots, "task-A", center)
        return detection_spots
#===============================================================================
# CustomConnectivityMatrix
#===============================================================================
class CustomConnectivityMatrix(ConnectivityMatrix):
    def __new__(cls, config:object=None, save:bool=True, force:bool=False):
        if config is not None:
            config.PATH_CONNECTIVITY = "custom" + config.PATH_CONNECTIVITY
            path = config.path_to_connectivity_matrix()
            if not force and PIC.path_exists(path):
                return PIC.load(path)
        return super().__new__(cls, force=force)


    def get_shift(self, config)->np.ndarray:
        shift = np.full((config.rows, config.rows), fill_value=None)
        shift[80:90, 55:65] = get_start(10)
        shift[80:90, 35:45] = get_start(10)
        shift[50:80, 35:65] = get_gate(30)
        shift[40:50, 45:55] = get_repeat(10, gap=6)
        shift[10:40, 35:65] = get_select(30)
        return shift.ravel()


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    config = TaskConfig()
    
    # plt.figure()
    # plt.imshow(get_select(20), origin="lower", cmap="jet")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(get_repeat(20, gap=12), origin="lower", cmap="jet")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(get_gate(20), origin="lower", cmap="jet")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(get_start(20), origin="lower", cmap="jet")
    # plt.colorbar()
    # return
    
    
    # conn = CustomConnectivityMatrix(config, force=True)
    conn = CustomConnectivityMatrix(config)
    
    simulator = BrianSimulator(config, conn)
    simulator.run_warmup()
    
    force = True
    force = False
    for seed in range(1):
        if force or not PIC.load_rate(config.baseline_tag(seed), sub_directory=config.sub_dir, config=config, dry=True):
            simulator.run_baseline(seed)
        
          
        force=True
        force=False      
        amount = config.AMOUNT_NEURONS[0]
        for taskname, center in config.task.items():
            tag_patch = UNI.get_tag_ident(taskname, config.radius[0], amount, 0, seed)
        
            if force or not PIC.load_rate(tag_patch, sub_directory=config.sub_dir, config=config, dry=True):
                
                for name, percent in center:
                    c = config.center_range[name]
                    dop_area = DOP.circular_patch(config.rows, c, config.radius[0])
                    patch = UNI.get_neurons_from_patch(dop_area, amount, seed+1)
                    logger.info(f"Neurons: {patch}")
                    # Update weights with different percentages
                    simulator.modulate_synapses(patch, percent)
                simulator.run_patch(tag_patch, seed, dop_patch=None, skip_reset=True)
    
    
    
    import pandas as pd
    merged = correlate_task_sequences(tag_patch, config, len(config.task["task-A"]))
    df = pd.DataFrame.from_dict(merged, orient='index')
    df.plot(kind='bar')

    return
    
    tag = config.baseline_tag(0)
    rate = PIC.load_rate(tag, exc_only=True, sub_directory=config.sub_dir, config=config)
    
    tag = tag_patch
    rate = PIC.load_rate(tag, exc_only=True, sub_directory=config.sub_dir, config=config)
    
    
    
    
    
    import analysis.dbscan_sequences as dbs
    scanner = dbs.DBScan_Sequences(config)
    spikes, labels = scanner._scan_spike_train(tag, force=force)
            
    from plot.sequences import plot_sequence_landscape
    plot_sequence_landscape(config.baseline_tags, config)
    ax = plot_sequence_landscape(tag_patch, config)
    
    from plot.lib import plot_patch 
    for name, c in config.center_range.items():
        plot_patch(c, config.radius[0], width=config.rows, axis=ax)
        ax.text(*c, name, verticalalignment="center", horizontalalignment="center", zorder=12)
    return


    from plot.animation import Animator
    from plot import AnimationConfig as figcfg
    animator = Animator(config, figcfg)
    animator.animate(config.baseline_tags[:1])
    plt.show()
    return
    plt.figure()
    plt.imshow(get_select(20), origin="lower", cmap="jet")
    plt.colorbar()
#===============================================================================
# METHODS
#===============================================================================

def correlate_task_sequences(tag_patch:str, config:object, no_patches:int) -> dict:
    """
    This is a simplified version of the sequence correlator as no sequence traverse the toroid.
    no_patches: The number of patches in the task
    """
    
    correlator = SequenceCorrelator(config)
    detection_spots = config.analysis.dbscan_controls.detection_spots_by_tag(tag_patch)
    correlator.count_shared_sequences(tag_patch, force_patch=True, force_baseline=True)
    df_bs = PIC.load_sequence_at_center(config.get_baseline_tag_from_tag(tag_patch), detection_spots, config)
    df = PIC.load_sequence_at_center(tag_patch, detection_spots, config)
    merged = {}
    for p in product([True, False], repeat=no_patches):
        tmp_bs = df_bs
        tmp    = df
        t = ""
        for c, column in enumerate(p):
            if column:
                tmp_bs = tmp_bs[tmp_bs[f"C{c}"]]
                tmp    = tmp[tmp[f"C{c}"]]
                t = t+f"&C{c}" if t else f"C{c}"
        if t == "":
            continue
        merged[t] = tmp_bs.shape[0], tmp.shape[0]
    return merged

#===============================================================================
# SHIFT METHODS
#===============================================================================

def get_select(rows:int):
    selectmotif = np.zeros((rows, rows))
    half = rows // 2

    selectmotif[:, :half] = 0
    selectmotif[:, half:] = -4

    # main path
    selectmotif[:, half-3:half]   = -1
    selectmotif[:, half  :half+3] = -3
    selectmotif[:, half-2:half+2] = -2
    

    # intersection
    selectmotif[half-1:half, half-4:half]   = -1
    selectmotif[half-1:half, half-2:half]   = -2
    selectmotif[half-1:half, half:half+4]   = -3
    selectmotif[half-1:half, half:half+2]   = -2

    ## Second alternative: Uncomment and set the start of the loop to 2
    selectmotif[half-2:half-1, half-5:half]   = -1
    selectmotif[half-2:half-1, half-3:half]   = -2
    selectmotif[half-2:half-1, half-1:half]   = -3
    
    selectmotif[half-2:half-1, half:half+5]   = -3
    selectmotif[half-2:half-1, half:half+3]   = -2
    selectmotif[half-2:half-1, half:half+1]   = -1
    
    selectmotif[half-3:half, half-3:half+3]   = -2

    for i in range(2, half):
        selectmotif[max(half-i-1, 0):half-i, max(half-i-6, 0):half]    = -1
        selectmotif[max(half-i-1, 0):half-i, max(half-i-2, 0):half]    = -2
        selectmotif[max(half-i-1, 0):half-i, max(half-i+1, 0):half]    = -3
        selectmotif[max(half-i-1, 0):half-i, max(half-i+3, 0):half]    = -4
        selectmotif[max(half-i-1, 0):half-i, max(half-i+4, 0):half]    = -5
        selectmotif[max(half-i-1, 0):half-i, max(half-i+6, 0):half]    = -6
        
        selectmotif[max(half-i-1, 0):half-i, half:min(half+i+6, rows)] = -3
        selectmotif[max(half-i-1, 0):half-i, half:min(half+i+2, rows)] = -2
        selectmotif[max(half-i-1, 0):half-i, half:min(half+i-1, rows)] = -1
        selectmotif[max(half-i-1, 0):half-i, half:min(half+i-3, rows)] =  0
        selectmotif[max(half-i-1, 0):half-i, half:min(half+i-4, rows)] =  1
        selectmotif[max(half-i-1, 0):half-i, half:min(half+i-6, rows)] =  2
    return selectmotif * np.pi/4



def get_start(rows):
    start = np.zeros((rows, rows))
    half = rows // 2

    start[:, :half] = 0
    start[:, half:] = -4
    
    start[half+4:half+8, half-1:half]   = -1
    start[half+4:half+8, half  :half+1] = -3
    
    start[:half+4, half-2:half]   = -1
    start[:half+4, half  :half+2] = -3
    
    # main path
    start[:half, half-3:half]   = -1
    start[:half, half  :half+3] = -3
    start[:half, half-1:half+1] = -2

    return start * np.pi/4

def get_repeat(rows, gap:int=2):
    repeat = np.zeros((rows, rows))
    half = rows // 2

    repeat[:, :half] = 0
    repeat[:, half:] = -4
    
    # main path
    repeat[:, half-3:half]   = -1
    repeat[:, half  :half+3] = -3
    repeat[:, half-1:half+1] = -2
    
    offset = gap // 2
    repeat[half-offset:half-offset+gap, half-4:half-1] = None
    repeat[half-offset:half-offset+gap, half+1:half+4] = None
        
    return repeat * np.pi/4


def get_gate(rows):
    gate = np.zeros((rows, rows))
    half = rows // 2

    gate[:, :half] = 0
    gate[:, half:] = 4

    # main path
    gate[:, half-3:half]   = -1
    gate[:, half  :half+3] = -3
    gate[:, half-1:half+1] = -2

    # intersection
    # gate[half-1:half, half-4:half]   = -1
    # gate[half-1:half, half-2:half]   = -2
    # gate[half-1:half, half:half+4]   = -3
    # gate[half-1:half, half:half+2]   = -2

    ## Second alternative: Uncomment and set the start of the loop to 2
    # gate[half-2:half-1, half-5:half]   = -1
    # gate[half-2:half-1, half-3:half]   = -2
    # gate[half-2:half-1, half-1:half]   = -3
    # gate[half-2:half-1, half:half+5]   = -3
    # gate[half-2:half-1, half:half+3]   = -2
    # gate[half-2:half-1, half:half+1]   = -1
        
    straight = 3
    for i in range(half-straight):
        gate[max(half+i, 0):half+i+1, max(half-i-4, 0):half]    =  1
        gate[max(half+i, 0):half+i+1, max(half-i-2, 0):half]    =  0
        gate[max(half+i, 0):half+i+1, max(half-i-1, 0):half]    = -1
        gate[max(half+i, 0):half+i+1, max(half-i+2, 0):half]    = -2
        gate[max(half+i, 0):half+i+1, max(half-i+3, 0):half]    = -3
        gate[max(half+i, 0):half+i+1, max(half-i+5, 0):half]    = -4
        
        gate[max(half+i, 0):half+i+1, half:min(half+i+4, rows)] =  3
        gate[max(half+i, 0):half+i+1, half:min(half+i+2, rows)] = -4
        gate[max(half+i, 0):half+i+1, half:min(half+i+1, rows)] = -3
        gate[max(half+i, 0):half+i+1, half:min(half+i-2, rows)] =  -2
        gate[max(half+i, 0):half+i+1, half:min(half+i-3, rows)] =  -1
        gate[max(half+i, 0):half+i+1, half:min(half+i-5, rows)] =  -2
        
    gate[-straight:, max(half-i-4, 0):half] =  0
    gate[-straight:, max(half-i-2, 0):half] = -1
    gate[-straight:, max(half-i-0, 0):half] = -2
    gate[-straight:, max(half-i+2, 0):half] = -3
    gate[-straight:, max(half-i+4, 0):half] = -4
    
    gate[-straight:, half:min(half+i+4, rows)] = -4
    gate[-straight:, half:min(half+i+2, rows)] = -3
    gate[-straight:, half:min(half+i+0, rows)] = -2
    gate[-straight:, half:min(half+i-2, rows)] = -1
    gate[-straight:, half:min(half+i-4, rows)] =  0
    
    return gate * np.pi/4

#===============================================================================
if __name__ == '__main__':
    main()
    plt.show()