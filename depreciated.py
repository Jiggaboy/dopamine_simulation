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






def analyze_travel_direction(patch:np.ndarray, patchdetails:tuple, postfix:str=None, delta_t:float=None, threshold:float=None, plot_rates:bool=True):
    threshold = threshold or 0.3
    delta_t = delta_t or 10

    # load rate
    rate = PIC.load_rate(postfix, skip_warmup=True, exc_only=True)

    mean_rate = rate[patch].mean(axis=0)
    # Check
    if plot_rates:
        RAT.rate(rate[patch], avg=True, threshold=threshold)

    # Get crossings above threshold and avoid IndexErrors by cutting results that would be after end of simulation.
    crossings = np.where(mean_rate > threshold)[0]
    crossings = crossings[crossings + delta_t < rate.shape[1]]
    snapshot_pre = rate[:, crossings].mean(axis=1)
    snapshot_post = rate[:, crossings + delta_t].mean(axis=1)

    title = f"Snapshot @{patchdetails[0]} with r={patchdetails[1]}"
    title_pre =title + f"\n No. threshold crossings: {crossings.size}"
    title_post = title + f"\n Delta t: {delta_t}ms"
    des =  {"title_pre": title_pre,
            "title_post": title_post,}

    ACT.pre_post_activity(snapshot_pre, snapshot_post, **des)
    
    
def analyze_anatomy():
    populations = []
    populations.append((Population.load(CF.SPACE_WIDTH), "b", "c"))
    populations.append((Population.load(CF.SPACE_WIDTH, terminated=True), "y", "orange"))

    populations[0][0].plot_population()
    neuron_base = np.random.randint(populations[0][0].exc_neurons.size)
    # neuron_base = 438
    # neuron_base = 2489
    print(f"Neuron: {neuron_base}")

    for population_set in populations:
        population = population_set[0]

        W = population.connectivity_matrix

        # Select neurons
        gridsize = 2
        total_neurons = gridsize**2
        neuron_slices = []
        for i in range(gridsize):
            right = neuron_base - (i * CF.SPACE_WIDTH) + 1
            left = right - gridsize
            if left % CF.SPACE_WIDTH > right % CF.SPACE_WIDTH:
                row = population.coordinates[right][1]
                neuron_slices.append(slice(left + CF.SPACE_WIDTH, (row + 1) * CF.SPACE_WIDTH))
                neuron_slices.append(slice(row * CF.SPACE_WIDTH, right))
            else:
                neuron_slices.append(slice(left, right))

        idcs = []
        for sl in neuron_slices:
            plt.scatter(*population.coordinates[sl].T, c="g")
            idx = range(*sl.indices(population.exc_neurons.size))
            idcs.extend(list(idx))
        start = idcs

        all_connected_neurons = set()
        for _ in range(100):
            condensed_W = W[idcs, :]
            condensed_out_degree = condensed_W.sum(axis=0)
            idcs = condensed_out_degree.argsort()[-total_neurons:][::-1]

            all_connected_neurons.update(idcs)

        # print(f"Starting neurons: {start}")
        # print(f"Finishing neurons: {idcs}")

        distances = []
        for i in range(total_neurons):
            for j in range(total_neurons):
                distance = population.grid.get_distance(population.coordinates[start[i]], population.coordinates[idcs[j]])
            distances.append(distance)
        print(np.mean(distances))

        plt.scatter(*population.coordinates[sorted(all_connected_neurons)].T, c=population_set[1])
        plt.scatter(*population.coordinates[start[0]].T, c="g")
        plt.scatter(*population.coordinates[idcs].T, c=population_set[2])
        plt.title("Neurons involved in aSTAS")