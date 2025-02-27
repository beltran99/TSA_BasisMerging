__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import numpy as np
import pandas as pd

import testbed
from combinatorics import get_set_partitions

from pathlib import Path
ROOT_DIR = Path(__file__).parent / '..'

def com(merge: list, new_centroid: int, b: dict, y: pd.DataFrame, bases_w: pd.DataFrame) -> float:
    """This function computes the CoM for a given merge of bases and a possible new centroid, given the RHS vector, the duals and the bases' weights.

    Args:
        merge (list): List of bases indices to be merged
        new_centroid (int): Index of the possible new centroid of the merge
        b (dict): RHS vector
        y (pd.DataFrame): Duals of the aggregated model
        bases (pd.DataFrame): Weights of the bases

    Returns:
        float: Cost of merging the bases in the merge list with the new centroid
    """        
    com = 0
    misclassified = [x for x in merge if x != new_centroid]
    for i in misclassified:
        com += np.matmul(b[i], y.loc[i] - (bases_w.loc[i] / bases_w.loc[new_centroid])*y.loc[new_centroid] )

    return com


if __name__ == "__main__":
    """
    This script computes the CoM for all possible bases mergers in the optimal transport problem case study.
    """
    # load input data
    df_cf = pd.read_excel(ROOT_DIR / 'data/opt_transport.xlsx', sheet_name='cap_factors')
    df_cf.drop(columns='generator', inplace=True)
    df_demand = pd.read_excel(ROOT_DIR / 'data/opt_transport.xlsx', sheet_name='demand')
    df_input = pd.merge(df_demand, df_cf, on='period')

    # solve full model to obtain duals and map input data to bases
    full_model, _ = testbed.run_case(config_path=ROOT_DIR / 'data/opt_transport.xlsx')
    df_duals = testbed.export_duals(full_model, ROOT_DIR / 'results/opt_transport')
    df_bases = testbed.identify_bases(df_duals)

    df_input.period = df_input.period.astype('int64')
    df_bases.period = df_bases.period.astype('int64')

    df = pd.merge(df_input, df_bases, on='period')

    # solve agg. model to obtain aggregated duals
    df_bases_centroids = df.groupby('basis').agg({'cap_factor': ['mean'], 'demand': ['mean'], 'weight': ['max']})
    df_bases_centroids.reset_index(drop=False, inplace=True)
    df_bases_centroids.columns = df_bases_centroids.columns.get_level_values(0)

    _, agg_config_path = testbed.generate_agg_config(reference='opt_transport.xlsx', df_centroids=df_bases_centroids)
    agg_model, w = testbed.run_case(config_path=agg_config_path)
    agg_duals = testbed.export_duals(agg_model, ROOT_DIR / 'results/opt_transport_agg')
    agg_duals.period = agg_duals.period.astype('int64')
    agg_duals.set_index('period', inplace=True)

    df_bases_centroids.set_index('basis', inplace=True)

    b = {
        basis: np.array(
            [df[df.basis == basis]['demand'].mean(), # eBalance_bus_1
             0, # eBalance_bus_2
             0, # eBalance_bus_3
             500, # eMaxProd_t1
             (df[df.basis == basis].cap_factor*500).mean(), # eMaxProd_w1
             df[df.basis == basis]['demand'].mean(), # eNSP
             500, # eMaxLim_1_exp
             500, # eMaxLim_1_imp
             250, # eMaxLim_2_exp
             250, # eMaxLim_2_imp
             150, # eMaxLim_3_exp
             150  # eMaxLim_3_imp
            ]
            ) for basis in df.basis.unique()
    }

    # obtain all possible partitions of the set of bases I
    set_size = 8
    s = list(range(1, set_size+1))
    partitions = get_set_partitions(s)

    for p in partitions:
        acc_com = 0
        for x in p:
            if len(x) > 1:  # for every element in a partition, compute the CoM if it is not a singleton
                x_com = dict(zip(s, len(s)*[0]))
                for possible_centroid in x_com.keys():  # compute the CoM for every possible centroid and take the minimum
                    x_com[possible_centroid] = com(x, possible_centroid, b, agg_duals, df_bases_centroids.weight)
                acc_com += min(x_com.values())
        print(f"Bases merger: {p}\tCoM: {acc_com}")