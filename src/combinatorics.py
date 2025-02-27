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

def get_set_partitions_with_adj(lst: list, adjacent_bases: list) -> list:
    """Generates all possible partitions of a set of bases with adjacency constraints between bases

    Args:
        lst (list): Set to be partitioned
        adjacent_bases (list): List of adjacent bases

    Returns:
        list: List of partitions
    """    

    def is_valid_merge(partition, adj_candidates):
        for subpartition in partition:
            if len(subpartition) > 1:
                if not np.any([set(subpartition) == set(x) for x in adj_candidates]):
                    return False
        return True
    
    # generate adjacent combinations
    def get_all_candidate_subsets(original_set, adjacent_bases):
        candidate_sets = adjacent_bases.copy()
        for _ in range(2, len(original_set)):
            prod = list(itertools.product(candidate_sets, candidate_sets))
            candidate_sets = [list(set(x[0] + x[1])) for x in prod]

            # remove duplicates
            new_candidate_sets = []
            for elem in candidate_sets:
                if elem not in new_candidate_sets:
                    new_candidate_sets.append(elem)
            candidate_sets = new_candidate_sets.copy()

        return candidate_sets
    adj_candidates = get_all_candidate_subsets(lst, adjacent_bases)
    
    # generate combinations
    combinations = []
    for partition in generate_partitions(lst):
        if is_valid_merge(partition, adj_candidates):
            combinations.append([x for x in partition if len(x) > 0])

    # sort combinations
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

def str_to_list(s):
    """Converts a string to a list of strings"""

    if type(s) == str and len(s) > 0:
        return re.sub(r"[\'\[\]\s]", "", s).split(',')
    else:
        return ""
    
def partition_list_to_df(partitions):
    """Converts a list of partitions to a pandas DataFrame"""

    partitions_str = []
    for merger in partitions:
        merger_str = []
        for element in merger:
            element = list(map(str, element))
            merger_str.append("+".join(element))
        partitions_str.append(str(merger_str))

    df = pd.DataFrame(partitions_str, columns=['merge_list'])
    df['n_basis'] = df['merge_list'].apply(lambda x: len(str_to_list(x)))
    df['merge_list'] = df['merge_list'].apply(lambda x: str_to_list(x))
    df['merge_set'] = df['merge_list'].apply(lambda row: [set(map(int, x.split('+'))) for x in row])

    return df

def greedy_selection(partitions_df: pd.DataFrame, initial_greedy_choice: set = set([1,2])) -> list:
    """This function computes the number of possible partitions for each level of aggregation using a greedy selection approach.
    This means that the given greedy choice is established for the initial merge, e.g., from 8 clusters to 7, and this choice is kept for all the subsequent merges in the different levels of aggregation, e.g., from 7 to 6, 6 to 5, etc.

    Args:
        partitions_df (pd.DataFrame): Pandas DataFrame with partitions
        initial_greedy_choice (set, optional): Merging decision for first level of aggregation, e.g., from 8 clusters to 7. Defaults to set([1,2]).

    Returns:
        list: Number of possible partitions for each level
    """

    n_partitions = []
    for level in range(partitions_df.n_basis.max(), 0, -1):
        count = 0
        if level == partitions_df.n_basis.max():
            count = 1
        elif level == partitions_df.n_basis.max()-1:
            count = math.comb(partitions_df.n_basis.max(), 2)
        else:
            _df = partitions_df[partitions_df.n_basis == level]
            for i, row in _df.iterrows():
                count += np.any([initial_greedy_choice.issubset(x) for x in row['merge_set']])
        n_partitions.append(count)
    return n_partitions


if __name__ == "__main__":

    print(f"Bell number")
    for set_size in range(8,0,-1):
        s = list(range(1, set_size+1))
        partitions = get_set_partitions(s)
        
        print(f"Set size: {set_size}\tNumber of partitions: {len(partitions)}")

    print(f"\nGreedy selection")
    for set_size in range(8, 0, -1):
        s = list(range(1, set_size+1))
        partitions = get_set_partitions(s)
        partitions_df = partition_list_to_df(partitions)
        greedy_partitions = greedy_selection(partitions_df)

        print(f"Set size: {set_size}\tNumber of partitions: {np.sum(greedy_partitions)}")

    print(f"\nGreedy selection with adjacency")
    original_set = [1,2,3,4,5,6,7,8]
    adj_bases = [[1,2], [1,4], [1,5], [1,8], [2,3], [2,4], [3,8], [4,7], [5,6], [5,7], [6,8]]
    for set_size in range(8, 0, -1):
        set_size_n_results = []
        if set_size == 1:
            set_size_n_results.append(1)
        elif set_size == 8:
            partitions_with_adj = get_set_partitions_with_adj(original_set, adjacent_bases=adj_bases)
            partitions_with_adj_df = partition_list_to_df(partitions_with_adj)

            for x,y in itertools.combinations(original_set, 2):
                if [x,y] in adj_bases:
                    counts = []
                    greedy = set([x,y])
                    for n in range(partitions_with_adj_df.n_basis.max(), 0, -1):
                        count = 0
                        if n == partitions_with_adj_df.n_basis.max():
                                count = 1
                        elif n == partitions_with_adj_df.n_basis.max()-1:
                            count = math.comb(partitions_with_adj_df.n_basis.max(), 2)
                        else:
                            _df = partitions_with_adj_df[partitions_with_adj_df.n_basis == n]
                            for i, row in _df.iterrows():
                                count += np.any([greedy.issubset(x) for x in row['merge_set']])
                        counts.append(count)
                    set_size_n_results.append(sum(counts))
        else:
            for set_j in itertools.combinations(original_set, set_size):
                set_j = list(set_j)
                partitions_with_adj = get_set_partitions_with_adj(set_j, adjacent_bases=adj_bases)
                partitions_with_adj_df = partition_list_to_df(partitions_with_adj)

                for x,y in itertools.combinations(set_j, 2):
                    if [x,y] in adj_bases:
                        counts = []
                        greedy = set([x,y])
                        for n in range(partitions_with_adj_df.n_basis.max(), 0, -1):
                            count = 0
                            if n == partitions_with_adj_df.n_basis.max():
                                count = 1
                            elif n == partitions_with_adj_df.n_basis.max()-1:
                                count = math.comb(partitions_with_adj_df.n_basis.max(), 2)
                            else:
                                _df = partitions_with_adj_df[partitions_with_adj_df.n_basis == n]
                                for i, row in _df.iterrows():
                                    count += np.any([greedy.issubset(x) for x in row['merge_set']])
                            counts.append(count)
                        set_size_n_results.append(sum(counts))

        print(f"Set size: {set_size}\tMin. number of partitions: {min(set_size_n_results)}\tMax. number of partitions: {max(set_size_n_results)}")
