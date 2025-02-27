__author__ = ["Beltran Castro Gomez"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Beltran Castro Gomez"]
__license__ = "MIT"
__maintainer__ = "Beltran Castro Gomez"

import os
import pyomo.environ as pyo
import pandas as pd

from pathlib import Path
PARENT_DIR = Path(__file__).parent
ROOT_DIR = Path(__file__).parent / '..'
SOLVER = 'gurobi'

def resolve_path(folder, file):
    """Auxiliary function to resolve the path of a file given a folder and a file name"""
    if (PARENT_DIR / folder / file).exists():
        return PARENT_DIR / folder / file
    elif (Path.cwd() / folder / file).exists():
        return Path.cwd() / folder / file
    else:
        for root, dirs, files in os.walk(ROOT_DIR):
            for name in files:
                if name == file:
                    return os.path.abspath(os.path.join(root, name))

def create_model(input_data:dict) -> pyo.ConcreteModel:
    """Defines Pyomo model for the optimal transport problem and creates an instance with the given input data (time series and parameters).

    Args:
        input_data (dict): Dictionary containing input data time series and system parameters.

    Returns:
        pyo.ConcreteModel: Pyomo model instance of the optimal transport problem.
    """    

    model = pyo.ConcreteModel(name="Optimal transport")

    ### Sets ###
    model.g = input_data['generators']
    model.t = pyo.Set(within = model.g, initialize = input_data['thermal_generators'])
    model.w = pyo.Set(within = model.g, initialize = input_data['wind_generators'])
    model.p = pyo.Set(initialize = input_data['periods'])
    model.l = pyo.RangeSet(len(input_data['line_limit']))

    ### Parameters ###
    # Cost parameters
    model.pVC_g = pyo.Param(model.g, initialize = input_data['vc_gen'])
    model.pVC_nsp = pyo.Param(initialize = input_data['vc_nsp'])
    model.pTransportCost = pyo.Param(initialize = input_data['cost_transport'])

    # Upper production limits of generators and line limits
    model.pMaxProd_g = pyo.Param(model.g, initialize = input_data['maxprod'])
    model.pLinLim_l = pyo.Param(model.l, initialize = input_data['line_limit'])
    
    # Input data time series
    model.pCapFac_w = pyo.Param(model.p, model.w, initialize = input_data['cf_wind'], domain=pyo.NonNegativeReals)
    model.pDemand_p = pyo.Param(model.p, initialize = input_data['demand'], domain=pyo.NonNegativeReals)

    # Period weights
    model.pWeight_p = pyo.Param(model.p, initialize = input_data['weight'], domain=pyo.NonNegativeReals)

    ### Variables ###
    model.vGen = pyo.Var(model.g, model.p, domain=pyo.NonNegativeReals)
    model.vNSP = pyo.Var(model.p, domain=pyo.NonNegativeReals)
    model.vLF = pyo.Var(((i, j, p) for i in model.l for j in model.l if i != j for p in model.p), domain=pyo.NonNegativeReals)

    ### Constraints ###
    # Upper production limits of generators
    def eMaxProd_rule(mdl, g, i):
        if g in mdl.w:
            return mdl.pMaxProd_g[g]*mdl.pCapFac_w[i, g] >= mdl.vGen[g, i]
        else:
            return mdl.pMaxProd_g[g] >= mdl.vGen[g, i]
    model.eMaxProd = pyo.Constraint(model.g, model.p, rule=eMaxProd_rule)

    # Non-supplied power limit
    def eMaxNSP(mdl, i):
        return mdl.pDemand_p[i] >= mdl.vNSP[i]
    model.eNSP = pyo.Constraint(model.p, rule=eMaxNSP)

    # Nodal balance equations
    def eBalance_bus_1(mdl, i):
        return mdl.pDemand_p[i] == - mdl.vLF[1, 3, i] - mdl.vLF[1, 2, i] + \
                mdl.vLF[3, 1, i] + mdl.vLF[2, 1, i] \
                + mdl.vNSP[i]
    model.eBalance_bus_1 = pyo.Constraint(model.p, rule=eBalance_bus_1)

    def eBalance_bus_2(mdl, i):
        return 0 == - mdl.vLF[2, 3, i] + mdl.vLF[1, 2, i] + \
               mdl.vLF[3, 2, i] - mdl.vLF[2, 1, i] + \
               mdl.vGen['t1', i]
    model.eBalance_bus_2 = pyo.Constraint(model.p, rule=eBalance_bus_2)

    def eBalance_bus_3(mdl, i):
        return 0 == mdl.vLF[2, 3, i] + mdl.vLF[1, 3, i] - \
               mdl.vLF[3, 1, i] - mdl.vLF[3, 2, i] + \
               mdl.vGen['w1', i]
    model.eBalance_bus_3 = pyo.Constraint(model.p, rule=eBalance_bus_3)

    # Maximum flow constraints
    def eMaxLim_1_exp_rule(mdl, i):
        return mdl.pLinLim_l[1] >= mdl.vLF[1, 2, i]
    model.eMaxLim_1_exp = pyo.Constraint(model.p, rule=eMaxLim_1_exp_rule)

    def eMaxLim_1_imp_rule(mdl, i):
        return mdl.pLinLim_l[1] >= mdl.vLF[2, 1, i]
    model.eMaxLim_1_imp = pyo.Constraint(model.p, rule=eMaxLim_1_imp_rule)

    def eMaxLim_2_exp_rule(mdl, i):
        return mdl.pLinLim_l[2] >= mdl.vLF[2, 3, i]
    model.eMaxLim_2_exp = pyo.Constraint(model.p, rule=eMaxLim_2_exp_rule)

    def eMaxLim_2_imp_rule(mdl, i):
        return mdl.pLinLim_l[2] >= mdl.vLF[3, 2, i]
    model.eMaxLim_2_imp = pyo.Constraint(model.p, rule=eMaxLim_2_imp_rule)

    def eMaxLim_3_exp_rule(mdl, i):
        return mdl.pLinLim_l[3] >= mdl.vLF[3, 1, i]
    model.eMaxLim_3_exp = pyo.Constraint(model.p, rule=eMaxLim_3_exp_rule)

    def eMaxLim_3_imp_rule(mdl, i):
        return mdl.pLinLim_l[3] >= mdl.vLF[1, 3, i]
    model.eMaxLim_3_imp = pyo.Constraint(model.p, rule=eMaxLim_3_imp_rule)

    ### Objective function ###
    def eCostNet_rule(mdl, i):
        return sum(mdl.vGen[g, i]*mdl.pVC_g[g] for g in mdl.g) + mdl.vNSP[i]*mdl.pVC_nsp + (sum(mdl.vLF[j, k, i]*mdl.pTransportCost for j in range(1, 4) for k in range(1, 4) if j != k))
    model.vCost = pyo.Expression(model.p, rule=eCostNet_rule)

    def eObjective_rule(mdl):
        return sum(mdl.vCost[i] * mdl.pWeight_p[i] for i in model.p)
    model.z = pyo.Objective(rule=eObjective_rule, sense=pyo.minimize)

    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return model

def data_load(config_path: str) -> dict:
    """Loads the data from the configuration excel file and returns it as a dictionary.

    Args:
        config_path (str): Configuration excel file path

    Returns:
        dict: Dictionary containing the input data from the configuration excel file.
    """    

    wsConfig = pd.read_excel(config_path, 'config')
    wsThermal = pd.read_excel(config_path, 'thermal', index_col=0)
    wsVRES = pd.read_excel(config_path, 'vres', index_col=0)
    wsDemand = pd.read_excel(config_path, 'demand', index_col=0)
    wsCapFactors = pd.read_excel(config_path, 'cap_factors', index_col=0)
    wsWeight = pd.read_excel(config_path, "weight", index_col=0)
    wsLineLimits = pd.read_excel(config_path, 'line_limits', index_col=0)

    input_data = dict()
    input_data['periods'] = list(map(str, list(pd.read_excel(config_path, "weight")['period'].values)))
    input_data['vc_nsp'] = wsConfig['vc_nsp'].values[0]
    input_data['cost_transport'] = wsConfig['cost_transport'].values[0]

    # get the data from the thermal generators
    the_gen = list(wsThermal.index)
    input_data['thermal_generators'] = the_gen

    # get the data from the wind generators
    wnd_gen = list(wsVRES.index)
    input_data['wind_generators'] = wnd_gen

    the_gen.extend(wnd_gen)
    input_data['generators'] = the_gen

    # use the data from thermal generators for its parameters
    input_data['maxprod'] = dict()
    input_data['vc_gen'] = dict()
    for idx, r in wsThermal.iterrows():
        input_data['maxprod'][idx] = r['installed_cap']
        input_data['vc_gen'][idx] = r['vc']

    # use the data from wind generators for its parameters
    for idx, r in wsVRES.iterrows():
        input_data['maxprod'][idx] = r['installed_cap']
        input_data['vc_gen'][idx] = r['vc']

    input_data['line_limit'] = dict()
    for idx, r in wsLineLimits.iterrows():
        input_data['line_limit'][idx] = r['limit']

    # get data for capacity factors, demand and weight
    input_data['demand'] = dict()
    input_data['cf_wind'] = dict()
    input_data['weight'] = dict()

    for idx, r in wsDemand.iterrows():
        input_data['demand'][str(idx)] = r['demand']

    for idx, r in wsCapFactors.iterrows():
        input_data['cf_wind'][(str(idx), r['generator'])] = r['cap_factor']

    for idx, r in wsWeight.iterrows():
        input_data['weight'][str(idx)] = r['weight']

    return input_data

def run_case(config_path = ROOT_DIR / 'data/opt_transport.xlsx', verbose = False) -> tuple:
    """Solves the optimal transport problem using the configuration excel file data.

    Args:
        config_path (_type_, optional): Configuration excel file containing input data to the optimal transport problem. Defaults to ROOT_DIR/'data/opt_transport.xlsx'.
        verbose (bool, optional): Verbosity option. Defaults to False.

    Returns:
        tuple: Instantiated Pyomo model and period weights
    """    

    data = data_load(config_path)
    model = create_model(data)
    solver = pyo.SolverFactory(SOLVER, solver_io="lp", tee=verbose)
    res = solver.solve(model)
    if verbose:
        print(res)

    return model, data['weight']

def idx_match(row:pd.Series, df:pd.DataFrame) -> list:
    """Returns the indexes of all rows in the DataFrame that match a given row."""

    idxs = []
    for idx, r in df.iterrows():
        if all(row == r):
            idxs.append(idx)

    return idxs

def export_solution(mdl: pyo.ConcreteModel, out_path: str, weights=None) -> tuple:
    """Exports the solution of the Pyomo model to an Excel file.

    Args:
        mdl (pyo.ConcreteModel): Optimal transport problem model instance.
        out_path (str): Output path to save the model solution.
        weights (_type_, optional): Dictionary containing period weights. Defaults to None.

    Returns:
        tuple: pd.DataFrame containing the absolute solution and pd.DataFrame containing the hourly solution.
    """    

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    w = 1

    sln = []

    sln_dict = dict()
    l_sln_hourly = []

    sln_dict['of_value'] = pyo.value(mdl.z)
    sln_dict['thermal'] = 0
    sln_dict['renewable'] = 0
    sln_dict['nsp'] = 0

    # this code is only for illustrative purposes as it is not
    # the most efficient way to obtain data from Pyomo
    for v in mdl.component_objects(ctype=pyo.Var):
        aux_var = {'var': str(v)}
        aux_d = {}
        for idx in v:
            aux_d[idx] = pyo.value(v[idx])
            if str(v) == 'vGen':
                g, p = idx
                if weights is not None:
                    w = weights[p]

                if g == 'w1':
                    sln_dict['renewable'] = sln_dict['renewable'] + aux_d[idx]*w
                else:
                    sln_dict['thermal'] = sln_dict['thermal'] + aux_d[idx]*w

            elif str(v) == 'vNSP':
                if weights is not None:
                    w = weights[idx]

                sln_dict['nsp'] = sln_dict['nsp'] + aux_d[idx]*w

            elif str(v) == 'vLF':
                l1, l2, p = idx
                if weights is not None:
                    w = weights[p]
                sln_dict[str(l1)+'_to_'+str(l2)] = sln_dict.get(str(l1)+'_to_'+str(l2), 0) + aux_d[idx]*w
            else:
                raise Exception("Unknown variable in model!")

        aux_var['values'] = aux_d

        sln.append(aux_var)

    for elem in sln:
        df = pd.DataFrame.from_dict(elem['values'], orient='index', columns=[str(elem['var'])])

        if all(isinstance(item, tuple) for item in df.index):
            df.index = pd.MultiIndex.from_tuples(df.index)

        df.reset_index(drop=False, inplace=True)

        excelfile = out_path / (str(elem['var']) + '.xlsx')
        df.to_excel(excelfile, index=False)

        if 'vGen' in df.columns:
            df = df.pivot(index='level_1', columns='level_0', values='vGen').reset_index()
            df.rename(columns={'level_1': 'period'}, inplace=True)
        elif 'vLF' in df.columns:
                df = df.pivot(index='level_2', columns=['level_0', 'level_1'], values='vLF').reset_index()
                df.rename(columns={'level_2': 'period'}, inplace=True)
                df.columns = ['period'] + ['_to_'.join(map(str, col)).strip() for col in df.columns.values[1:]]
        else:
            df.rename(columns={'index': 'period'}, inplace=True)
        l_sln_hourly.append(df)

    df_hourly = None
    if len(l_sln_hourly) > 0:
        df_hourly = l_sln_hourly[0]
        for i in range(1, len(l_sln_hourly)):
            df_hourly = df_hourly.merge(right=l_sln_hourly[i], on='period')

    df_complete = pd.DataFrame.from_dict(sln_dict, orient='index').rename(columns={0: 'complete'})
    excelfile = out_path / 'results.xlsx'
    df_complete.to_excel(excelfile)

    return df_complete, df_hourly

def export_model_comparison(df_full: pd.DataFrame, df_aggregated: pd.DataFrame) -> pd.DataFrame:
    """Exports the comparison between the full model and the aggregated model to an Excel file.

    Args:
        df_full (pd.DataFrame): Full model solution
        df_aggregated (pd.DataFrame): Aggregated model solution

    Returns:
        pd.DataFrame: Comparison between the full model and the aggregated model
    """    

    df = df_full.copy()
    df.insert(1, 'aggregated', df_aggregated['complete'].values)
    
    df['delta'] = df['complete'] - df['aggregated']
    df['rel_delta'] = 1 - df['aggregated'] / df['complete']
    df['rel_delta'].fillna(0, inplace=True)

    return df

def export_duals(model: pyo.ConcreteModel, out_path: str) -> pd.DataFrame:
    """Exports the duals of the Pyomo model to an Excel file.

    Args:
        model (pyo.ConcreteModel): Optimal transport problem model instance.
        out_path (str): Output path to save the duals.

    Returns:
        pd.DataFrame: DataFrame containing the duals of the Pyomo model.
    """    

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    aux_d = {}
    for k, v in model.dual.items():
        aux_name = k.parent_component().local_name
        if isinstance(k.index(), tuple):
            aux_name = aux_name + "_" + pyo.value(k.index()[0])

        try:
            aux_d[aux_name]
        except KeyError as e:
            aux_d[aux_name] = {}
        if isinstance(k.index(), tuple):
            aux_d[aux_name][pyo.value(k.index()[1])] = v
        else:
            aux_d[aux_name][pyo.value(k.index())] = v

    df_duals = pd.DataFrame.from_dict({key: aux_d[key] for key in ['eBalance_bus_1', 'eBalance_bus_2', 'eBalance_bus_3']}, orient='columns').reset_index(drop=False)

    for k, v in aux_d.items():
        if not 'eBalance' in k:
            df_aux = pd.DataFrame.from_dict(aux_d[k], orient='index').reset_index(drop=False).rename(columns={0: k})
            df_duals = pd.merge(left=df_duals, right=df_aux, left_on='index', right_on='index', how='left')

    df_duals.rename(columns={'index': 'period'}, inplace=True)
    df_duals.fillna(0, inplace=True)

    df_duals.to_excel(out_path / "duals.xlsx", index=False)

    return df_duals

def export_of_values(model_run: pyo.ConcreteModel) -> list:
    """Exports the hourly objective function values of the Pyomo model to a list."""

    of_values = []
    for of_value in model_run.vCost:
        of_values.append(pyo.value(model_run.vCost[of_value]))
    
    return of_values

def generate_agg_config(reference: str, df_centroids: pd.DataFrame, config_dir = None, config_name = None) -> tuple:
    """Generates the configuration file for an aggregated model based on the full model parameter data and the given input data centroids.

    Args:
        reference (str): Path to the reference configuration file
        df_centroids (pd.DataFrame): DataFrame containing the input data centroids of the clusters
        config_dir (_type_, optional): Output directory for the configuration file. Defaults to None.
        config_name (_type_, optional): Name of the configuration file. Defaults to None.

    Returns:
        tuple: pd.DataFrame containing the configuration data and the path to the configuration file.
    """    

    reference_path = resolve_path('../data', reference)

    if config_name is None:
        config_name = reference.split('.xlsx')[0] + '_agg.xlsx'
    else:
        config_name = config_name + '.xlsx'

    if config_dir is None:
        config_dir = PARENT_DIR / str('../data/' + config_name)
    else:
        config_dir = PARENT_DIR / config_dir / config_name

    df_config = pd.read_excel(reference_path, sheet_name=None)

    # config sheet
    df_config['config']['periods'] = [len(df_centroids)]

    # demand sheet
    df_config['demand'] = pd.DataFrame({'period': df_centroids['basis'], 'demand': df_centroids['demand']})

    # cap_factors sheet
    df_config['cap_factors'] = pd.DataFrame({'period': df_centroids['basis'], 'generator': 'w1', 'cap_factor': df_centroids['cap_factor']})

    # weight sheet
    df_config['weight'] = pd.DataFrame({'period': df_centroids['basis'], 'weight': df_centroids['weight']})

    with pd.ExcelWriter(config_dir) as writer:
        for sheet_name, df in df_config.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return df_config, config_dir

def identify_bases(df_duals) -> pd.DataFrame:
    """Identifies the bases of the optimal transport problem based on the duals of the full problem and retrieves the mapping of time steps and bases."""

    df_duals['period'] = df_duals['period'].astype(int)
    df_duals.sort_values(by='period', inplace=True)
    df_duals['period'] = df_duals['period'].astype(str)
    df_duals.reset_index(drop=True)

    _df = df_duals.copy()
    _df['basis'] = ''
    _df['weight'] = 0

    bases = df_duals.iloc[:, 1:].drop_duplicates().reset_index(drop=True)
    df_duals_aux = df_duals.drop(columns='period')

    i = 1
    for _, r in bases.iterrows():
        idx_basis = idx_match(r, df_duals_aux)
        _df.loc[idx_basis, 'basis'] = i
        _df.loc[idx_basis, 'weight'] = len(idx_basis)
        i += 1

    return _df[['period', 'basis', 'weight']]