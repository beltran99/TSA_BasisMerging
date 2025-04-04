__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2025, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

from com import compute_min_com_merger
from combinatorics import get_set_partitions


def exhaustive_strategy(partitions: list, bases_set: list) -> None:
    """This function implements the Exhaustive strategy for merging bases.

    Args:
        partitions (list): All possible partitions of the set of bases.
        bases_set (list): The original set of bases.
    """
    set_size = len(bases_set)
    c = 0

    for level in range(set_size, 0, -1):

        mergers = [x for x in partitions if len(x) == level]
        merger, merger_com = compute_min_com_merger(bases_set, mergers)

        print(f'Partitions of size {level}. Number of possible partitions: {len(mergers)}')
        print(f'Possible partitions: {mergers}')
        print(f'Merger choice: {merger}')
        print(f'Updated set: {s}')
        print()

        c += len(mergers)


if __name__ == "__main__":

    print(f"\nExhaustive strategy\n")

    set_size = 8
    s = list(range(1, set_size+1))
    partitions = get_set_partitions(s)
    exhaustive_strategy(partitions, s)
    
    print(f"Set size: {set_size}. Number of partitions: {len(partitions)}\n\n")