__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2025, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import itertools

from com import compute_min_com_merger

def greedy_strategy(s: list):
    """This function implements the Greedy strategy for merging bases.

    Args:
        s (list): Set of bases

    Returns:
        int: Total number of evaluated mergers
    """    
    s_int = list(map(int, s))
    set_size = len(s)
    c = 1
    for level in range(set_size, 1, -1):

        mergers_tmp = list(itertools.combinations(s, 2))
        mergers_str = ['_'.join(i for i in x) for x in mergers_tmp]
        mergers_int = [[list(map(int, x.split('_')))] for x in mergers_str]

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

        print(f'Aggregation from {level} to {level - 1} clusters. Number of possible mergers: {len(mergers_tmp)}')
        print(f'Possible mergers: {mergers_tmp}')
        print(f'Greedy choice: {greedy_choice}')
        print(f'Updated set: {s}')
        print()

        c += len(mergers_tmp)

    return c


if __name__ == "__main__":

    print(f"\nGreedy strategy\n")

    set_size = 8
    s = list(range(1, set_size+1))
    s = list(map(str, s))
    c = greedy_strategy(s)
    
    print(f"Set size: {set_size}. Number of partitions: {c}\n\n")