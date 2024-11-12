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

import matplotlib.pyplot as plt
import numpy as np


#===============================================================================
# MAIN METHOD
#===============================================================================
def main():
    # Example Usage
    grid = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1]
    ])
    
    cluster = find_cluster(grid)
    print("Cluster:", cluster)


#===============================================================================
# METHODS
#===============================================================================

def find_cluster(grid):
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    
    if grid.size == 0:
        return []

    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    cluster = []

    def dfs(r, c, island):
        # Check bounds and if cell is already visited or water
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r, c] or grid[r][c] == 0:
            return
        # Mark the cell as visited and part of the island
        visited[r, c] = True
        island.append((r, c))
        # Visit all 4 possible directions
        dfs(r - 1, c, island)  # Up
        dfs(r + 1, c, island)  # Down
        dfs(r, c - 1, island)  # Left
        dfs(r, c + 1, island)  # Right

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r, c]:
                island = []
                dfs(r, c, island)
                cluster.append(np.asarray(island))

    return cluster


def get_cluster_dimensions(*cluster) -> type:
    xy_dimensions = np.zeros((len(cluster), 2))
    for ci, c in enumerate(cluster):
        xy_dimensions[ci, :] = c.max(axis=0) - c.min(axis=0) + 1
    return xy_dimensions

#===============================================================================
if __name__ == '__main__':
    main()
