__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import itertools

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
