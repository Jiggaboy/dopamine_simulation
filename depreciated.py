#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:03:13 2021

@author: hauke
"""



def update_synaptic_weights(rate:np.ndarray, t:int, population:Population, excOnly:bool=True, dopamineAreasOnly:bool=False)->np.ndarray:
    INTERVAL = 50
    ETA = 1
    conn_matrix = population.connectivity_matrix
    if t-INTERVAL >= 0 and t % INTERVAL == 0:
        W_temp = conn_matrix.copy()
        if excOnly:
            W_temp[W_temp<0] = 0
            W_temp[W_temp>0] = 1

        if dopamineAreasOnly:
            for area in diffusion_areas:
                for neuron in range(len(population.exc_neurons)):
                    if  population.grid.get_distance(area.pos, population.coordinates[neuron]) > area.radius:
                        W_temp[:, neuron] = 0.

        correlation = np.corrcoef(rate[:, t-INTERVAL: t])
        # if True: # only pos. correlation
        #     correlation[correlation < 0] = 0
        conn_matrix = conn_matrix + ETA * W_temp * correlation
        # print(t, conn_matrix)
        # conn_matrix = conn_matrix + conn_matrix * np.abs(correlation)
    return conn_matrix
