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
from skimage.morphology import max_tree
import networkx as nx
from dataclasses import dataclass

from params import config
import lib.universal as UNI
import lib.pickler as PIC

tag = config.baseline_tag(seed=0)
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================
def main():
    tags = None
    if isinstance(tag, (list, tuple)):
        tags = tag
        name = UNI.name_from_tag(tags[0])
        radius = UNI.radius_from_tag(tags[0])
    else:
        name = UNI.name_from_tag(tag)
        radius = UNI.radius_from_tag(tag)

    # Single tag vs multiple tags (then averaged)
    if not tags:
        seq_count = _get_sequence_landscape(tag, config)
    else:
        seq_counts = []
        for t in tags:
            seq_count_tmp = _get_sequence_landscape(t, config)
            seq_counts.append(seq_count_tmp)
        seq_count = np.asarray(seq_counts, dtype=int).mean(axis=0)

    plt.imshow(seq_count.T, origin="lower", cmap="hot_r")



    from lib import dfs
    from dataclasses import field
    from typing import List, Dict

    @dataclass
    class Tree:
        _id: int
        levels: Dict = field(default_factory=dict)


        def __str__(self):
            return f"{self._id}: {self.levels.keys()}"

        def __repr__(self):
            return f"{self._id}: {self.levels.keys()}"


        def add_level(self, threshold:int, leafs: np.ndarray) -> None:
            assert threshold not in self.levels.keys()
            self.levels[threshold] = [leafs] # nested lists to enable merges


        def isin(self, leaf:int, threshold:int) -> bool:
            for branch in self.levels[threshold]:
                if leaf in branch:
                    return True
            return False


        def merge(self, tree:object):
            # Merge the other tree into this tree
            for threshold, leafs in tree.levels.items():
                self.levels[threshold].append(leafs)


    @dataclass
    class Forest:
        _id: int = 0
        trees: List = field(default_factory=list)

        def __iter__(self):
            for tree in self.trees:
                yield tree

        def __str__(self):
            return f"{self._id}: {[t._id for t in self.trees]}"

        def __repr__(self):
            return f"{self._id}: {[t._id for t in self.trees]}"

        def add_tree(self) -> Tree:
            tree = Tree(self._id)
            self.trees.append(tree)
            self._id += 1
            return tree


    forest = Forest()
    merges = []
    T = np.max(seq_count)
    for t in range(T, 0, -1):
        print(t)
        bin_seq_count = seq_count >= t

        clusters = dfs.find_cluster(bin_seq_count)
        # print(clusters)
        for c in clusters:
            parents = []
            # Check whether parts of the cluster was already found at a higher threshold
            for tree in forest:
                if t+1 in tree.levels.keys():
                    c_view    = c.view([('', c.dtype)] * c.shape[1])
                    leafs = tree.levels[t+1][0]
                    leaf_view = leafs.view([('', leafs.dtype)] * leafs.shape[1])
                    if np.isin(c_view, leaf_view).any():
                        parents.append(tree)
            if len(parents) == 1:
                parents[0].add_level(t, c)
            elif len(parents) > 1:
                root_id, root = np.inf, None
                for p in parents:
                    root_id = p._id if p._id < root_id else root_id
                    root    = p if p._id == root_id else root
                for p in parents:
                    if p._id == root_id:
                        root.add_level(t, c)
                        continue
                    root.merge(p)
                for p in parents:
                    if p._id != root_id:
                        merges.append((t, (root_id, p._id)))
                        # TODO: Delete trees from forest?
            # If the cluster has no parents, grow a new tree.
            else:
                logger.info("Grow new tree...")
                tree = forest.add_tree()
                tree.add_level(t, c)


    return
    ####### MAX TREE: GPT ####################################

    from scipy.ndimage import maximum_filter

    # 8-connected neighbors
    NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]



    # Union-Find
    class UF:
        def __init__(self, n):
            self.parent = list(range(n))
            self.height = [-np.inf] * n  # merge height
            self.peak = [-1] * n


        # Find the parent of the UF instance
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x


        def union(self, x, y, merge_h):
            rx = self.find(x)
            ry = self.find(y)
            if rx == ry: return rx
            # merge trees: store merge height
            self.parent[ry] = rx
            self.height[rx] = merge_h
            return rx


    def build_merge_tree(h):
        H, W = h.shape
        idx = lambda r, c: r*W + c

        # flatten and sort
        coords = [(h[r,c], r, c) for r in range(H) for c in range(W) if h[r,c] > 0]
        coords.sort(reverse=True)  # high → low

        uf = UF(H*W)
        active = np.zeros((H,W), dtype=bool)

        merges = []  # (merged_cluster1, merged_cluster2, height)

        for height, r, c in coords:
            active[r,c] = True
            this = idx(r,c)

            neighbors = []
            for dr,dc in NEIGHBORS:
                rr,cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W and active[rr,cc]:
                    neighbors.append(idx(rr,cc))

            # If no neighbors: it's a new peak
            if not neighbors:
                uf.peak[this] = height
                continue

            # Otherwise merge with neighbors
            for n in neighbors:
                root_before = uf.find(this)
                root_neighbor = uf.find(n)
                if root_before != root_neighbor:
                    uf.union(root_before, root_neighbor, height)
                    merges.append((root_before, root_neighbor, height))

        return uf, merges

    uf, merges = build_merge_tree(seq_count)
    print(merges)
    return


    ####### MAX TREE: START ####################################
    P, S = max_tree(seq_count)
    P_rav = P.ravel()
    seq_rav = seq_count.ravel()
    print(P.shape, S.shape)
    print(np.unique(P).size, np.unique(S))
    return

    plt.figure("parent")
    plt.imshow(P, origin="lower")


    # the canonical max-tree graph
    canonical_max_tree = nx.DiGraph()
    canonical_max_tree.add_nodes_from(S)
    for node in canonical_max_tree.nodes():
        canonical_max_tree.nodes[node]['value'] = seq_rav[node]
    canonical_max_tree.add_edges_from([(n, P_rav[n]) for n in S[1:]])

    # max-tree from the canonical max-tree
    nx_max_tree = nx.DiGraph(canonical_max_tree)
    labels = {}
    prune(nx_max_tree, S[0], labels)

    # component tree from the max-tree
    labels_ct = {}
    total = accumulate(nx_max_tree, S[0], labels_ct)

    # positions of nodes : canonical max-tree (CMT)
    pos_cmt = position_nodes_for_max_tree(canonical_max_tree, seq_rav)

    # positions of nodes : max-tree (MT)
    pos_mt = dict(zip(nx_max_tree.nodes, [pos_cmt[node] for node in nx_max_tree.nodes]))

    # plot the trees with networkx and matplotlib
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 8))

    # plot_tree(
    #     nx_max_tree,
    #     pos_mt,
    #     ax1,
    #     seq_rav,
    #     title='Component tree',
    #     labels=labels_ct,
    #     font_size=6,
    #     text_size=8,
    # )

    plot_tree(nx_max_tree, pos_mt, ax2, seq_rav, title='Max tree', labels=labels)

    plot_tree(canonical_max_tree, pos_cmt, ax3, seq_rav, title='Canonical max tree')


#===============================================================================
# METHODS
#===============================================================================



def _get_sequence_landscape(tag:str, config:object):
    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = np.zeros(shape=(config.rows, config.rows), dtype=int)
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        spike_set = spikes[labels == label]
        spike_set = np.unique(spike_set[:, 1:], axis=0).T
        seq_count[tuple(spike_set)] += 1
    return seq_count




def plot_tree(graph, positions, ax, seq_rav, *, title='', labels=None, font_size=8, text_size=8):
    """Plot max and component trees."""
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=ax,
        node_size=40,
        node_shape='s',
        node_color='white',
        font_size=font_size,
        labels=labels,
    )
    for v in range(seq_rav.min(), seq_rav.max() + 1):
        ax.hlines(v - 0.5, -3, 10, linestyles='dotted')
        ax.text(-3, v - 0.15, f"val: {v}", fontsize=text_size)
    ax.hlines(v + 0.5, -3, 10, linestyles='dotted')
    ax.set_xlim(-3, 10)
    ax.set_title(title)
    ax.set_axis_off()

def prune(G, node, res):
    """Transform a canonical max tree to a max tree."""
    value = G.nodes[node]['value']
    res[node] = str(node)
    preds = [p for p in G.predecessors(node)]
    for p in preds:
        if G.nodes[p]['value'] == value:
            res[node] += f", {p}"
            G.remove_node(p)
        else:
            prune(G, p, res)
    G.nodes[node]['label'] = res[node]
    return


def accumulate(G, node, res):
    """Transform a max tree to a component tree."""
    total = G.nodes[node]['label']
    parents = G.predecessors(node)
    for p in parents:
        total += ', ' + accumulate(G, p, res)
    res[node] = total
    return total


def position_nodes_for_max_tree(G, image_rav, root_x=4, delta_x=1.2):
    """Set the position of nodes of a max-tree.

    This function helps to visually distinguish between nodes at the same
    level of the hierarchy and nodes at different levels.
    """
    pos = {}
    canonical_max_tree = G
    for node in reversed(list(nx.topological_sort(canonical_max_tree))):
        value = G.nodes[node]['value']
        if canonical_max_tree.out_degree(node) == 0:
            # root
            pos[node] = (root_x, value)

        in_nodes = [y for y in canonical_max_tree.predecessors(node)]

        # place the nodes at the same level
        level_nodes = [y for y in filter(lambda x: image_rav[x] == value, in_nodes)]
        nb_level_nodes = len(level_nodes) + 1

        c = nb_level_nodes // 2
        i = -c
        if len(level_nodes) < 3:
            hy = 0
            m = 0
        else:
            hy = 0.25
            m = hy / (c - 1)

        for level_node in level_nodes:
            if i == 0:
                i += 1
            if len(level_nodes) < 3:
                pos[level_node] = (pos[node][0] + i * 0.6 * delta_x, value)
            else:
                pos[level_node] = (
                    pos[node][0] + i * 0.6 * delta_x,
                    value + m * (2 * np.abs(i) - c - 1),
                )
            i += 1

        # place the nodes at different levels
        other_level_nodes = [
            y for y in filter(lambda x: image_rav[x] > value, in_nodes)
        ]
        if len(other_level_nodes) == 1:
            i = 0
        else:
            i = -len(other_level_nodes) // 2
        for other_level_node in other_level_nodes:
            if (len(other_level_nodes) % 2 == 0) and (i == 0):
                i += 1
            pos[other_level_node] = (
                pos[node][0] + i * delta_x,
                image_rav[other_level_node],
            )
            i += 1

    return pos



if __name__ == '__main__':
    main()
    plt.show()
