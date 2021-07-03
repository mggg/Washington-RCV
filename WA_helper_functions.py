import random
import matplotlib.pyplot as plt
from gerrychain.tree import PopulatedGraph, random_spanning_tree, find_balanced_edge_cuts_memoization




def gdf_print_map(partition, filename, gdf, unit_name):
    cdict = {partition.graph.nodes[i][unit_name]:partition.assignment[i] for i in partition.graph.nodes()}
    gdf['color'] = gdf.apply(lambda x: cdict[x[unit_name]], axis=1)
    plt.figure()
    gdf.plot(column='color', cmap = 'tab10')
    plt.savefig(filename+'.png', dpi = 600)
    plt.close("all")


def capped_tries_bipartition_tree(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    spanning_tree_fn=random_spanning_tree,
    balance_edge_fn=find_balanced_edge_cuts_memoization,
    choice=random.choice,
    max_tries = 1000
):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = spanning_tree_fn(graph)
    restarts = 0
    tries = 0
    while len(possible_cuts) == 0 and tries < max_tries:
        if restarts == node_repeats:
            spanning_tree = spanning_tree_fn(graph)
            restarts = 0
            tries += 1
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = balance_edge_fn(h, choice=choice)
        restarts += 1
    if tries == max_tries:
        return set()
    return choice(possible_cuts).subset