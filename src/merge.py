__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import testbed

from pathlib import Path
PARENT_DIR = Path(__file__).parent
ROOT_DIR = Path(__file__).parent / '..'

class BasesMerger:
    """
        Interface to merge bases and choose bases mergers
    """

    def __init__(self, input_data, bases_mapping) -> None:
        """ Initialization of a Bases Merger object given some time series input data.

        Args:
            input_data (pd.Dataframe): input data (period, demand, cap_factor)
            bases_mapping (pd.Dataframe): bases mapping (period, basis)
        """

        df = pd.merge(input_data, bases_mapping, on='period')
        
        self.data = df.copy()
        self.basis = sorted(self.data.basis.unique())

        self.colors_iter = iter(sns.color_palette("bright"))
        self.basis_colors = {}
        for basis in self.basis:
            c = next(self.colors_iter)
            self.basis_colors[int(basis)] = c
            self.basis_colors[str(basis)] = c
    
    def merge(self, merger: list, plot_merge: bool = False) -> tuple:
        """Executes an aggregated model based on a bases merger.

        Args:
            merger (list): Bases merger with the following format: [[1], [2, 3], [4], [5], [6], [7], [8]].
            plot_merge (bool, optional): Whether to plot resulting input space after the merge. Defaults to False.

        Returns:
            tuple: Config DataFrame of the aggregated model, general results of the aggregated model, and hourly results of the aggregated model.
        """        
        df_merge = self.data.copy()
        df_merge['basis'] = df_merge.basis.astype('str')
        
        # re-label merged bases, e.g., '2' and '3' to '2_3'
        for cluster in merger:
            cluster = list(map(str, cluster))
            new_label = "_".join(cluster)
            for element in cluster:
                df_merge.basis.replace(element, new_label, inplace=True)

        df_merge['weight'] = df_merge.groupby('basis')['basis'].transform("count")
    
        # compute centroids of the bases merger
        df_bases_centroids = df_merge.groupby('basis').agg({'cap_factor': ['mean'], 'demand': ['mean'], 'weight': ['max']})
        df_bases_centroids.reset_index(drop=False, inplace=True)
        df_bases_centroids.columns = df_bases_centroids.columns.get_level_values(0)

        case_name = ""
        for element in merger:
            element = list(map(str, element))
            case_name += '-' + "_".join(element)
        case_name = 'merger' + case_name

        data_dir = ROOT_DIR / 'data/merge'
        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir = ROOT_DIR / 'results/merge' / case_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # generate and execute the aggregated model
        df_config, config_path = testbed.generate_agg_config('opt_transport.xlsx', df_bases_centroids, config_dir=data_dir, config_name=case_name)
        model, weights = testbed.run_case(config_path=config_path)
        df_agg_ext, df_hourly_agg = testbed.export_solution(model, results_dir, weights=weights)
        _ = testbed.export_duals(model, results_dir)

        if plot_merge:
            self.visualize_merge(df_merge, df_bases_centroids)

        return df_config, df_agg_ext, df_hourly_agg
    
    def read_config_data(self, config_path: str = ROOT_DIR / 'data/opt_transport.xlsx'):
        """Auxiliary function to read configuration data from case study."""

        config = pd.read_excel(config_path, sheet_name='config')
        vres = pd.read_excel(config_path, sheet_name='vres')
        thermal = pd.read_excel(config_path, sheet_name='thermal')
        lines = pd.read_excel(config_path, sheet_name='line_limits')

        maxcap_w = vres['installed_cap'].values[0]
        maxcap_th = thermal['installed_cap'].values[0]
        maxcap_l1 = lines[lines.line == 1]['limit'].values[0]
        maxcap_l2 = lines[lines.line == 2]['limit'].values[0]
        maxcap_l3 = lines[lines.line == 3]['limit'].values[0]

        vc_w = vres['vc'].values[0]
        vc_th = thermal['vc'].values[0]
        vc_nsp = config['vc_nsp'].values[0]
        vc_line = 0.1

        return dict(
            maxcap_w = maxcap_w,
            maxcap_th = maxcap_th,
            maxcap_l1 = maxcap_l1,
            maxcap_l2 = maxcap_l2,
            maxcap_l3 = maxcap_l3,
            vc_w = vc_w,
            vc_th = vc_th,
            vc_nsp = vc_nsp,
            vc_line = vc_line,
        )
    
    def get_basis(self, demand, cap_factor):
        """Function to determine the basis of a given point from input data demand and capacity factor based on if-else rules."""

        config = self.read_config_data()

        maxcap_w = config['maxcap_w']
        maxcap_th = config['maxcap_th']
        maxcap_l1 = config['maxcap_l1']
        maxcap_l2 = config['maxcap_l2']
        maxcap_l3 = config['maxcap_l3']

        if (demand > cap_factor*maxcap_w and demand <= cap_factor*maxcap_w + maxcap_th) and (cap_factor*maxcap_w >= maxcap_l3 and cap_factor*maxcap_w - maxcap_l3 < maxcap_l2 and demand - maxcap_l3 < maxcap_l1):
            return 1
        elif (demand <= cap_factor*maxcap_w) and (demand >= maxcap_l3 and demand - maxcap_l3 < maxcap_l2):
            return 2
        elif (demand <= cap_factor*maxcap_w) and (demand < maxcap_l3):
            return 3
        elif (demand >= maxcap_l2 + maxcap_l3) and (demand < maxcap_l2 + maxcap_l3 + maxcap_th) and (cap_factor*maxcap_w >= maxcap_l2 + maxcap_l3) and (demand - maxcap_l3 < maxcap_l1):
            return 4
        elif (demand > maxcap_l1 + maxcap_l3) and (cap_factor*maxcap_w > maxcap_l3) and (cap_factor*maxcap_w < maxcap_l3+maxcap_l2) and (demand - maxcap_l3 > maxcap_l1) and (cap_factor*maxcap_w - maxcap_l3 < maxcap_l2):
            return 5
        elif (demand > maxcap_th + cap_factor*maxcap_w) and (cap_factor*maxcap_w <= maxcap_l3) and (demand - cap_factor*maxcap_w > maxcap_l1):
            return 6
        elif (demand > maxcap_l1 + maxcap_l3) and (cap_factor*maxcap_w > maxcap_l3) and (cap_factor*maxcap_w - maxcap_l3 > maxcap_l2):
            return 7
        elif (demand <= cap_factor*maxcap_w + maxcap_th) and (cap_factor*maxcap_w < maxcap_l3):
            return 8
    
    def visualize_merge(self, df_merge: pd.DataFrame, df_basis_centroids: pd.DataFrame) -> None:
        """Function to visualize input space after the merge.

        Args:
            df_merge (pd.DataFrame): DataFrame with bases merger input data
            df_basis_centroids (pd.DataFrame): DataFrame with bases merger centroids
        """        

        _df_merge = df_merge.copy()
        _df_basis_centroids = df_basis_centroids.copy()
        _df_merge.basis = _df_merge.basis.apply(lambda x: x.replace('_', '+'))
        _df_basis_centroids.basis = _df_basis_centroids.basis.apply(lambda x: x.replace('_', '+'))

        colors = self.basis_colors.copy()
        for _, row in _df_basis_centroids.iterrows():
            if row['basis'] not in colors:
                color = self.get_basis(row['demand'], row['cap_factor'])
                colors[row['basis']] = self.basis_colors[color]

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.scatterplot(data=_df_merge, x="demand", y="cap_factor", hue="basis", palette=colors, ax=ax)
        sns.scatterplot(data=_df_basis_centroids, x="demand", y="cap_factor", marker="X", color='black', s=100, ax=ax)

        ax.legend(loc='lower right', bbox_to_anchor=(1.20, 0), title='basis')

        plt.xlabel('Demand [MW]')
        plt.ylabel('Wind capacity factor')
        plt.tight_layout()

        plt.show()
    
