__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import pandas as pd

import testbed
from combinatorics import get_set_partitions
from merge import BasesMerger

from pathlib import Path
ROOT_DIR = Path(__file__).parent / '..'

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    """
    This scripts generates all possible bases mergers from the set of bases I found in the optimal transport problem and solves all the corresponding aggregated models to compare them with the full model.
    """
    
    # load input data
    df_cf = pd.read_excel(ROOT_DIR / 'data/opt_transport.xlsx', sheet_name='cap_factors')
    df_cf.drop(columns='generator', inplace=True)
    df_demand = pd.read_excel(ROOT_DIR / 'data/opt_transport.xlsx', sheet_name='demand')
    df_input = pd.merge(df_demand, df_cf, on='period')
    df_input.period = df_input.period.astype(str)

    # solve full model to obtain duals and map input data to bases
    full_model, _ = testbed.run_case(config_path=ROOT_DIR / 'data/opt_transport.xlsx')
    df_full, _ = testbed.export_solution(full_model, 'results/opt_transport')
    df_duals = testbed.export_duals(full_model, ROOT_DIR / 'results/opt_transport')
    df_bases = testbed.identify_bases(df_duals)
    df_bases.period = df_bases.period.astype(str)

    # generate all possible mergers
    set_size = 8
    s = list(range(1, set_size+1))
    mergers = get_set_partitions(s)
    n_mergers = len(mergers)

    counter = 0
    results = []

    bases_merger = BasesMerger(df_input[['period', 'demand', 'cap_factor']].copy(), df_bases)

    # solve aggregated models and save results
    for merger in mergers:
        counter += 1
        print(f'Solving bases merger {counter}/{n_mergers}: {merger}')
        
        _, df_merge_results, _ = bases_merger.merge(merger)
        df_comparison = testbed.export_model_comparison(df_full, df_merge_results)

        cols = ['merger']
        cols.extend([f'error_{result}' for result in df_comparison.index])
        cols.extend([f'rel_error_{result}' for result in df_comparison.index])

        merger_str = []
        for element in merger:
            element = list(map(str, element))
            merger_str.append("+".join(element))

        result = [merger_str]
        result.extend(list(df_comparison.delta.values))
        result.extend(list(df_comparison.rel_delta.values))
        results.append(result)

    results_df = pd.DataFrame(columns=cols, data=results)
    results_df.to_csv(ROOT_DIR / 'merger_enumeration.csv', index=False)