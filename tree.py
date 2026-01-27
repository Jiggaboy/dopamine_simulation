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
from plot.sequences import _get_sequence_landscape

_tag = config.baseline_tag(seed=1)
# _tag = [config.baseline_tag(seed=0), *config.get_all_tags("repeat-main", seeds=0)]
# _tag = [*config.get_all_tags("repeat-main", seeds=0)]
#===============================================================================
# MAIN METHOD AND TESTING AREA
#===============================================================================

def main():
    
    fig, ax = plt.subplots(num="indegree")
    from lib.connectivitymatrix import ConnectivityMatrix
    conn = ConnectivityMatrix(config)
    indegree, _ = conn.degree(conn._EE)
    indegree = indegree * config.synapse.weight
    ax.set_title("In-degree")
    H, edges, _ = ax.hist(indegree.flatten(), bins=15, weights=np.ones(len(indegree.flatten())) / len(indegree.flatten()) * 100)

    #### Aim: Get the number of merges per baseline simulation
    # seeds = np.arange(300, 340)
    # seeds = np.asarray([config.landscape.params["base"]])
    seeds = np.arange(1)
    merge_counter = np.zeros(seeds.size)
    for s, seed in enumerate(seeds):
    # for s, base in enumerate(seeds):
        # config.landscape.params["base"] = base
        tag = config.baseline_tag(seed=seed)
    
        spikes, labels = PIC.load_spike_train(tag, config)
        seq_count = _get_sequence_landscape(spikes, labels, config.rows)
    
        forest, merges, bridge_neurons = grow_forest(seq_count)
        merge_counter[s] = len(merges)
    
        plt.figure(tag)
        plt.imshow(seq_count.T, origin="lower", cmap="hot_r")
        plt.scatter(*bridge_neurons.T, c="cyan", marker="*")
    

        
        bridge_neurons = bridge_neurons.astype(int)
        bridge_degrees = indegree[bridge_neurons[:, 1], bridge_neurons[:, 0]]
        ax.hist(bridge_degrees, edges, weights=np.ones(len(bridge_degrees)) / len(indegree.flatten()) * 100, fill=False)
        
        plt.figure(f"Bridge on indegree {s}")
        degree_cmap = plt.cm.jet
        im = plt.imshow(indegree,
                        origin="lower",
                        cmap=degree_cmap,
        )
        plt.scatter(*bridge_neurons.T, c="black", marker="*", zorder=10)
        
        conn = ConnectivityMatrix(config)
        indegree, _ = conn.degree(conn._EE)
        indegree = indegree * config.synapse.weight
    
        avgRate = PIC.load_average_rate(tag, sub_directory=config.sub_dir, config=config)
    
        plt.figure()
        plt.scatter(indegree.flatten(), avgRate)
        idx = bridge_neurons[:, 1] * config.rows + bridge_neurons[:, 0]
        plt.scatter(indegree.flatten()[idx], avgRate[idx], marker="*", c="lime")
        separator = np.linspace(indegree.min(), indegree.max(), 5+1)
        from figure_generator.figure2 import map_indegree_to_color
        for sep in separator[1:-1]:
            color = map_indegree_to_color(sep)
            ax.axvline(sep, ymax=0.8, ls="--", c=color)
    
    plt.figure("Hist of merges")
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
        # spikes, labels = PIC.load_spike_train(_tag, config)
        # seq_count = _get_sequence_landscape(spikes, labels, config.rows)
    # else:
    #     seq_counts = []
    #     for t in tags:
        # spikes, labels = PIC.load_spike_train(t, config)
        # seq_count_tmp = _get_sequence_landscape(spikes, labels, config.rows)
    #         seq_counts.append(seq_count_tmp)
    #     seq_count = np.asarray(seq_counts, dtype=int).mean(axis=0)

    # plt.imshow(seq_count.T, origin="lower", cmap="hot_r")
    
    
    for tag in tags:
        spikes, labels = PIC.load_spike_train(tag, config)
        seq_count = _get_sequence_landscape(spikes, labels, config.rows)


        forest, merges, _ = grow_forest(seq_count)
        
        G = nx.DiGraph()

        G.add_nodes_from([t._id for t in forest.trees])
        levels = {tree._id: list(tree.levels.keys())[0] for tree in forest.trees}

        extra_nodes = []
        edges = []
        merged_replacements = {}
        for merge in merges:
            print(merge)
            level, (root_node, branch_node), intersection = merge
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


def grow_forest(counts:np.ndarray):
    forest = Forest()
    merges = []
    bridge_neurons = np.zeros((0, 2))
    T = np.max(counts)
    for t in range(T, 0, -1):
        # Binarize the sequence count
        bin_counts = counts >= t

        # Requires the max recursion depth to be greater than 6400 (for a 80x80 network)
        clusters = dfs.find_cluster(bin_counts)
        
        for c in clusters:
            if c.shape[0] <= 3:
                continue
            c_view    = c.view([('', c.dtype)] * c.shape[1])
            
            parents = []
            # Check whether parts of the cluster was already found at a higher threshold
            for tree in forest:
                if t+1 in tree.levels.keys(): # if there could be a parent node
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
                
                union = []
                for p in parents:
                    union.extend(p.levels[t+1][0])
                    if p._id == root_id:
                        root.add_level(t, c)
                        continue
                    root.merge(p)
                union = np.asarray(union)
                union_view = union.view([('', union.dtype)] * union.shape[1])
                intersect_view = ~np.isin(c_view, union_view)
                intersect_idx = intersect_view.squeeze()
                intersect_coords = c[intersect_idx]
                for p in parents:
                    if p._id != root_id:
                        merges.append((t, (root_id, p._id), intersect_coords))
                        
                
                from lib.dopamine import circular_patch
                coordinates = UNI.get_coordinates(config.rows)
                mask = np.zeros(intersect_coords.shape[0], dtype=bool)
                mask_member = np.zeros(intersect_coords.shape[0], dtype=bool)
                for i, n in enumerate(intersect_coords):
                    patch = circular_patch(config.rows, n, radius=6) # radius of 6 or 7 is reasonable
                    vicinity = coordinates[patch]
                    vicinity_view = vicinity.view([('', vicinity.dtype)] * vicinity.shape[1])
                    patch_max = circular_patch(config.rows, n, radius=10) # radius of 6 or 7 is reasonable
                    vicinity_max = coordinates[patch_max]
                    vicinity_max_view = vicinity_max.view([('', vicinity_max.dtype)] * vicinity_max.shape[1])
                    found = np.zeros(len(parents))
                    found_member = np.zeros(len(parents))
                    for j, p in enumerate(parents):
                        parent_neurons = p.levels[t+1][0]
                        parent_view = parent_neurons.view([('', parent_neurons.dtype)] * parent_neurons.shape[1])
                        core_view = np.isin(vicinity_view, parent_view)
                        member_view = np.isin(vicinity_max_view, parent_view)
                        if np.any(core_view):
                            found[j] = True
                            found_member[j] = True
                    mask[i] = True if np.count_nonzero(found) >= 2 else False
                    mask_member[i] = True if np.count_nonzero(found_member) >= 2 else False
                print()
                # Task:
                # Find those neurons that bridge (intersect_coords[mask])
                # With those neurons, find all clusters they form -> neighbors are also bridge_neurons
                # The whole set is available in intersect_coords
                grid = np.zeros((config.rows, config.rows))
                grid[intersect_coords[mask_member][:, 0], intersect_coords[mask_member][:, 1]] = 1
                islands = dfs.find_cluster(grid)
                inter_view = intersect_coords[mask].view([('', intersect_coords[mask].dtype)] * intersect_coords[mask].shape[1])
                tmp_bridge = []
                for island in islands:
                    island_view = island.view([('', island.dtype)] * island.shape[1])
                    
                    member_view = np.isin(inter_view, island_view)
                    if np.any(member_view):
                        tmp_bridge.extend(island)
                tmp_bridge = np.asarray(tmp_bridge)
                # bridge_neurons = np.append(bridge_neurons, intersect_coords[mask], axis=0) 
                if np.any(tmp_bridge):
                    bridge_neurons = np.append(bridge_neurons, tmp_bridge, axis=0)      
                
                # plt.figure(t)
                # plt.imshow(counts.T, origin="lower", cmap="hot_r")
                # plt.scatter(*c.T, c="lightblue")
                # plt.scatter(*intersect_coords.T, c="green")
                # # plt.scatter(*intersect_coords[mask].T, c="blue", marker="*")  
                # # plt.scatter(*intersect_coords[~mask].T, c="lime", marker="*")      
                # plt.scatter(*tmp_bridge.T, c="white", marker=".")       
                
                # from sklearn.cluster import DBSCAN
                # db = DBSCAN(eps=2, min_samples=4)
                # db.fit(intersect_coords)
                # for l in np.unique(db.labels_):
                #     if l < 0:
                #         continue
                #     plt.scatter(*intersect_coords[db.labels_ == l].T, c="purple")
                # plt.show()
                # quit()
                print()
                        # TODO: Delete trees from forest?
            # If the cluster has no parents, grow a new tree.
                # plt.show()
            else:
                logger.info("Grow new tree...")
                tree = forest.add_tree()
                tree.add_level(t, c)
                
    bridge_neurons = bridge_neurons.astype(int)
    return forest, merges, bridge_neurons





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




if __name__ == '__main__':
    main()
    plt.show()
