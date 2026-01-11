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

_tag = config.baseline_tag(seed=0)
# _tag = [config.baseline_tag(seed=0), *config.get_all_tags("repeat-main", seeds=0)]
# _tag = [*config.get_all_tags("repeat-main", seeds=0)]
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def hierarchy_pos_custom_levels(
        G,
        root,
        level_attr="level",
        width=1.0,
        vert_scale=1.,
        x_start:float=0.,
    ):
        pos = {}
        subtree_width = {}

        def compute_subtree_width(node):
            children = list(G.successors(node))
            if not children:
                subtree_width[node] = 1
            else:
                subtree_width[node] = sum(compute_subtree_width(c) for c in children)
            return subtree_width[node]

        def assign_positions(node, x_left):
            level = G.nodes[node][level_attr]
            w = subtree_width[node]
            x_center = x_left + w / 2
            pos[node] = (x_center, level * vert_scale)

            x = x_left
            for child in G.successors(node):
                assign_positions(child, x)
                x += subtree_width[child]

        compute_subtree_width(root)
        assign_positions(root, x_start)

        return pos


def main():
    
    #### Aim: Get the number of merges per baseline simulation
    seeds = np.arange(300, 340)
    merge_counter = np.zeros(seeds.size)
    for s, base in enumerate(seeds):
        config.landscape.params["base"] = base
        tag = config.baseline_tag(seed=1)
    
        seq_count = _get_sequence_landscape(tag, config)
    
        forest, merges = grow_forest(seq_count)
        merge_counter[s] = len(merges)
        print(merges)
    plt.figure()
    plt.hist(merge_counter, bins=np.arange(25))
    
    return


    if isinstance(_tag, (list, tuple)):
        tags = _tag
        name = UNI.name_from_tag(tags[0])
        radius = UNI.radius_from_tag(tags[0])
    else:
        tags = [_tag]
        name = UNI.name_from_tag(_tag)
        radius = UNI.radius_from_tag(_tag)

    # Single tag vs multiple tags (then averaged)
    # if len(tags) == 1:
    #     seq_count = _get_sequence_landscape(_tag, config)
    # else:
    #     seq_counts = []
    #     for t in tags:
    #         seq_count_tmp = _get_sequence_landscape(t, config)
    #         seq_counts.append(seq_count_tmp)
    #     seq_count = np.asarray(seq_counts, dtype=int).mean(axis=0)

    # plt.imshow(seq_count.T, origin="lower", cmap="hot_r")
    # return
    
    
    for tag in tags:
        seq_count = _get_sequence_landscape(tag, config)
        plt.figure(tag)
        plt.imshow(seq_count.T, origin="lower", cmap="hot_r")


        forest, merges = grow_forest(seq_count)
        

        # Visualisation
        # import networkx as nx
        # G = nx.DiGraph()
        # G.add_nodes_from([t._id for t in forest.trees])

        import networkx as nx

        G = nx.DiGraph()

        G.add_nodes_from([t._id for t in forest.trees])
        levels = {tree._id: list(tree.levels.keys())[0] for tree in forest.trees}

        extra_nodes = []
        edges = []
        merged_replacements = {}
        for merge in merges:
            print(merge)
            level, (root_node, branch_node) = merge
            merge_node = f"{root_node} ({level})"
            if merge_node not in extra_nodes:
                extra_nodes.append(merge_node)


            if root_node in merged_replacements.keys():
                if merge_node != merged_replacements[root_node]:
                    edges.append((merge_node, merged_replacements[root_node]))
            else:
                if merge_node != root_node:
                    edges.append((merge_node, root_node))

            if branch_node in merged_replacements.keys():
                if merge_node != merged_replacements[branch_node]:
                    edges.append((merge_node, merged_replacements[branch_node]))
            else:
                if merge_node != branch_node:
                    edges.append((merge_node, branch_node))

            merged_replacements[root_node] = merge_node
            # edges.append((root_node, merge_node))
            # edges.append((branch_node, merge_node))
            levels[merge_node] = level
        G.add_nodes_from(extra_nodes)
        G.add_edges_from(edges)


        nx.set_node_attributes(G, levels, "level")

        leafs = forest.get_leafs()
        pos = {}
        xshift = 0
        for l, leaf in enumerate(leafs):
            pos_tmp = hierarchy_pos_custom_levels(G, merged_replacements.get(leaf._id, leaf._id), x_start=xshift)
            pos.update(pos_tmp)
            xshift = np.asarray(list(pos_tmp.values()), dtype=float)[:, 0].max() + 2
        # pos = hierarchy_pos_custom_levels_dag(G)
        plt.figure(f"{tag} with {len(merges)} merges")
        ax = plt.gca()
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=400,
            ax=ax
        )
        ax.set_axis_on()
        ax.yaxis.set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)


        # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # ax.set_ylim(-1, 21)   # adjust to your level range
        ax.set_ylabel("merge level (sequence count)")
        ax.set_yticks([0, 5, 10, 15, 20])
        print(ax.yaxis.get_visible(), ax.spines["left"].get_visible())
    #####################################################################
    # Bar-Plot: Start
    #####################################################################
    return

#===============================================================================
# METHODS
#===============================================================================


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


    def get_leafs(self):
        return [tree for tree in self.trees if 1 in tree.levels.keys()]


def _get_sequence_landscape(tag:str, config:object):
    spikes, labels = PIC.load_spike_train(tag, config)
    seq_count = np.zeros(shape=(config.rows, config.rows), dtype=int)
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        spike_set = spikes[labels == label]
        spike_set = np.unique(spike_set[:, 1:], axis=0).T
        seq_count[tuple(spike_set)] += 1
    return seq_count


def grow_forest(counts:np.ndarray):
    forest = Forest()
    merges = []
    T = np.max(counts)
    for t in range(T, 0, -1):
        # Binarize the sequence count
        bin_counts = counts >= t

        # Requires the max recursion depth to be greater than 6400 (for a 80x80 network)
        clusters = dfs.find_cluster(bin_counts)
        
        for c in clusters:
            parents = []
            # Check whether parts of the cluster was already found at a higher threshold
            for tree in forest:
                if t+1 in tree.levels.keys():
                    c_view    = c.view([('', c.dtype)] * c.shape[1])
                    leafs = tree.levels[t+1][0] # the first entry always contains all the previous nodes
                    leaf_view = leafs.view([('', leafs.dtype)] * leafs.shape[1])
                    if np.isin(c_view, leaf_view).any(): # Views are required to check equality across an axis
                        parents.append(tree)
                        
            # 1 parent -> same cluster; 0 parent -> new tree; more parents -> merge the pathways
            if len(parents) == 1:
                parents[0].add_level(t, c)
            elif len(parents) > 1:
                root_id, root = np.inf, None
                for p in parents:
                    root_id = p._id if p._id < root_id  else root_id
                    root    = p     if p._id == root_id else root   # root_id is updated first
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
    return forest, merges



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
