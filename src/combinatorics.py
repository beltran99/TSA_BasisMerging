__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import itertools
import re
import numpy as np
import pandas as pd
import math
import networkx as nx

def generate_partitions(lst):
    """ This function partitions a given set into non-empty subsets. It recursively adds each element to an existing subset or starts a new subset.

    Args:
        lst (_type_): List of elements to be partitioned
    """

    def partitions_helper(lst, start):
        if start == len(lst):
            yield [[]]
            return
        
        for subpartition in partitions_helper(lst, start + 1):
            yield subpartition
            for i, partition in enumerate(subpartition):
                yield subpartition[:i] + [partition + [lst[start]]] + subpartition[i+1:]
            yield subpartition + [[lst[start]]]

    for partition in partitions_helper(lst, 0):
        if len([item for sublist in partition for item in sublist]) == len(lst):
            yield partition

def get_set_partitions(lst: list) -> list:
    """Generates all possible partitions of a set

    Args:
        lst (list): Set to be partitioned

    Returns:
        list: List of partitions
    """

    # generate partitions
    combinations = []
    for partition in generate_partitions(lst):
        combinations.append([x for x in partition if len(x) > 0])

    # sort partitions
    sorted_combinations = []
    for combination in combinations:
        sorted_combination = []
        for element in combination:
            element.sort()
            sorted_combination.append(element)
        sorted_combination.sort()
        sorted_combinations.append(sorted_combination)

    sorted_combinations.sort()
    final_combinations = list(k for k,_ in itertools.groupby(sorted_combinations))

    return final_combinations

def greedy_strategy(s: list):
    set_size = len(s)
    for level in range(set_size, 1, -1):
        mergers = list(itertools.combinations(s, 2))
        mergers = ['_'.join(x) for x in mergers]

        # greedy selection
        # greedy_choice = next(iter(mergers))
        greedy_choice = np.random.choice(len(mergers))
        greedy_choice = mergers[greedy_choice]

        # update set s
        s = [x for x in s if x not in greedy_choice] + [greedy_choice]

        print(f'Aggregation from {level} to {level - 1}. Number of partitions: {len(mergers)}')
        print(f'Possible mergers: {mergers}')
        print(f'Greedy choice: {greedy_choice}')
        print(f'Updated set: {s}')
        print()

def is_valid_adj_merge(sets, unique_elements):

    sets = [tuple(map(int, x.split('_'))) for x in sets]
    unique_elements = list(map(int, unique_elements))
    unique_elements = set(unique_elements)

    G = nx.Graph()
    G.add_edges_from(sets)
    
    connected_components = list(nx.connected_components(G))[0]

    return connected_components == unique_elements

def greedy_adj_strategy(s: list, adj_bases = [[1,2], [1,4], [1,5], [1,8], [2,3], [2,4], [3,8], [4,7], [5,6], [5,7], [6,8]]):
    set_size = len(s)
    previous_s = s.copy()
    adj_bases_str = ['_'.join(map(str, x)) for x in adj_bases]

    for level in range(set_size, 1, -1):
        mergers = list(itertools.combinations(s, 2))
        mergers = ['_'.join(x) for x in mergers]
        
        # filter mergers
        filtered_mergers = []
        for merger in mergers:
            merger = list(sorted(merger.split('_')))

            if len(merger) == 2:
                if '_'.join(merger) in adj_bases_str:
                    filtered_mergers.append(merger)
            if len(merger) > 2:
                aux = [sorted(x) for x in itertools.combinations(merger, 2)]
                aux = ['_'.join(x) for x in aux]
                filtered_aux = [x for x in aux if x in adj_bases_str]
                if is_valid_adj_merge(filtered_aux, merger):
                    filtered_mergers.append(merger)

        # greedy selection
        if len(filtered_mergers) == 0:
            print(f'No valid mergers found. Exiting...')
            break
        
        # greedy_choice = next(iter(filtered_mergers))
        greedy_choice = np.random.choice(len(filtered_mergers))
        greedy_choice = filtered_mergers[greedy_choice]
        greedy_choice_str = '_'.join(greedy_choice)

        # update set s
        s = [greedy_choice_str]
        for element in previous_s:
            element_set = set(element.split('_'))
            gr_set = set(greedy_choice)
            if not element_set.issubset(gr_set):
                s.append(element)

        print(f'\nAggregation from {level} to {level - 1}. Number of partitions: {len(filtered_mergers)}')
        print(f'Valid mergers: {filtered_mergers}')
        print(f'Greedy choice: {greedy_choice}')
        print(f'Updated set: {s}')
        print()

        previous_s = s.copy()


if __name__ == "__main__":

    print(f"Bell number")
    for set_size in range(8,0,-1):
        s = list(range(1, set_size+1))
        partitions = get_set_partitions(s)
        print(f"Set size: {set_size}\tNumber of partitions: {len(partitions)}")

    print(f"\nGreedy selection")
    set_size = 8
    s = list(range(1, set_size+1))
    s = list(map(str, s))
    greedy_strategy(s)

    print(f"\nGreedy & Adjacency selection")
    set_size = 8
    s = list(range(1, set_size+1))
    s = list(map(str, s))
    greedy_adj_strategy(s)
