__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2025, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import itertools
import networkx as nx

from com import compute_min_com_merger

def is_valid_adj_merge(sets, unique_elements):
    """This function checks if the a merger of size greater than 2 staisfies the adjacency condition."""
    sets = [tuple(map(int, x.split('_'))) for x in sets]
    unique_elements = list(map(int, unique_elements))
    unique_elements = set(unique_elements)

    G = nx.Graph()
    G.add_edges_from(sets)
    
    connected_components = list(nx.connected_components(G))[0]

    return connected_components == unique_elements

def filter_mergers(mergers, adj_bases = [[1,2], [1,4], [1,5], [1,8], [2,3], [2,4], [3,8], [4,7], [5,6], [5,7], [6,8]], verbose=False):
    """This function filters the mergers based on the adjacency conditon."""
    adj_bases_str = ['_'.join(map(str, x)) for x in adj_bases]

    filtered_mergers = []

    for merger in mergers:
        merger = list(sorted(merger.split('_')))

        if len(merger) == 2:
            if '_'.join(merger) in adj_bases_str:
                filtered_mergers.append(merger)
                if verbose:
                    print(f"Valid merger: {merger}")
            elif verbose:
                print(f"Invalid merger: {merger}")
        if len(merger) > 2:
            aux = [sorted(x) for x in itertools.combinations(merger, 2)]
            aux = ['_'.join(x) for x in aux]
            filtered_aux = [x for x in aux if x in adj_bases_str]
            if is_valid_adj_merge(filtered_aux, merger):
                filtered_mergers.append(merger)
                if verbose:
                    print(f"Valid merger: {merger}")
            elif verbose:
                print(f"Invalid merger: {merger}")

    return filtered_mergers

def greedy_adj_strategy(s: list, adj_bases : list = [[1,2], [1,4], [1,5], [1,8], [2,3], [2,4], [3,8], [4,7], [5,6], [5,7], [6,8]], verbose : bool =True) -> tuple:
    """This function implements the Greedy & Adjacent strategy for merging bases.

    Args:
        s (list): The original set of bases
        adj_bases (list, optional): The list of pairs of adjacent bases. Defaults to [[1,2], [1,4], [1,5], [1,8], [2,3], [2,4], [3,8], [4,7], [5,6], [5,7], [6,8]].
        verbose (bool, optional): Verbosity option. Defaults to True.

    Returns:
        tuple: Total number of evaluated mergers and exit status.
    """    
    s_int = list(map(int, s))
    set_size = len(s)
    c = 1
    for level in range(set_size, 1, -1):

        mergers_tmp = list(itertools.combinations(s, 2))
        mergers_str = ['_'.join(i for i in x) for x in mergers_tmp]

        # filter adjacent mergers
        filtered_mergers = filter_mergers(mergers_str, adj_bases, verbose=verbose)
        if len(filtered_mergers) == 0:
            return c, 'EARLY EXIT'
        mergers_int = [[[int(i) for i in x]] for x in filtered_mergers]

        # greedy selection
        min_merger, min_com = compute_min_com_merger(s_int, mergers_int)
        greedy_choice = '_'.join(str(i) for i in min_merger[0])
        greedy_choice_set = set(list(map(int, greedy_choice.split('_'))))

        # update set s
        s_set = [set(list(map(int, x.split('_')))) for x in s]
        for element in s_set[:]:
            if element.issubset(greedy_choice_set):
                s_set.remove(element)
        s_set = s_set + [greedy_choice_set]

        s = ['_'.join(map(str, list(x))) for x in s_set]

        if verbose:
            print(f'\nAggregation from {level} to {level - 1} clusters. Number of possible mergers: {len(filtered_mergers)}')
            print(f'Possible mergers: {filtered_mergers}')
            print(f'Greedy choice: {greedy_choice}')
            print(f'Updated set: {s}')
            print()

        c += len(filtered_mergers)

    return c, 'CONVERGENCE'


if __name__ == "__main__":

    print(f"\nGreedy & Adjacency strategy\n")

    set_size = 8
    s = list(range(1, set_size+1))
    s = list(map(str, s))
    c, status = greedy_adj_strategy(s, verbose=True)

    print(f"Set size: {set_size}. Set: {s}. Number of partitions: {c}. Exit status: {status}")