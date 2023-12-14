"""
Root System Library

A library of Approximate Bayesian Computation methods for synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Internal 
from root_system_lib.stats import exec_root_stats_map

##########################################################################################################
### Library
##########################################################################################################


# From https://github.com/BayesianModelingandComputationInPython/BookCode_Edition1/blob/main/notebooks/scripts/rf_selector.py

def select_model(models: dict, obs_statistics: dict, root_stats_map: dict, root_stats: list, 
    kwargs_map: dict, df_cols: list, root_type: str = None, n_samples = 10, n_trees = 100, 
    f_max_features = 0.5, random_seed: int = None):
    """
    Use random forest to select the optimal Bayesian model.

    Args:
        models (dict):  
            A dictionary of tuples composed of Bayesian models and their respective traces.
        obs_statistics (dict): 
            A dictionary of precomputed summary statistics.
        root_stats_map (dict): 
            A dictionary of available root statistics.
        root_stats (list): 
            A list of requested summary statistics.
        kwargs_map (dict): 
            A dictionary of arguments to pass to the summary statistics functions.
        df_cols (list): 
            A list of column names to convert the array into a dataframe.
        root_type (str, optional): 
            A categorical label for root type. Defaults to None.
        n_samples (int, optional): 
            The number of posterior draws. Defaults to 10.
        n_trees (int, optional): 
            The number of trees within the random forest ensemble. Defaults to 100.
        f_max_features (float, optional): 
            The maximum number of features for random sub-sampling. Defaults to 0.5.
        random_seed (int, optional): 
            The random seed for replicability. Defaults to None.

    Returns:
        tuple: 
            The best performing model and its assocaited probability.
    """
    root_statistics = list(models.keys())
    n_models = len(root_statistics)
    ref_table = []

    # Build reference table
    # Table dimensionality = (models * n_samples, n_statistics)
    for model_results in models.values():
        model, trace = model_results
        obs_name = model.observed_RVs[0].name
        
        for _ in range(n_samples):
            pps = pm.sample_posterior_predictive(trace, samples = 1, size = None, model = model,
                                                progressbar = False)
            sim_df = pd.DataFrame(pps[obs_name].squeeze(), columns = df_cols)
            sim_statistics = exec_root_stats_map(sim_df, root_stats_map, root_stats, kwargs_map, root_type)
            sample_stats_list = []
            for stat in sim_statistics.values():
                if isinstance(stat, np.ndarray):
                    for v in stat:
                        sample_stats_list.append(v)
                else:
                    sample_stats_list.append(stat)

            ref_table.append(np.array(sample_stats_list))

    ref_table = np.array(ref_table)

    # Flatten summary statistics into vector
    obs_sum = []
    for stat in root_statistics:
        val = obs_statistics[stat]
        if val.ndim > 1:
            for v in val.T:
                obs_sum.append(v)
        else:
            obs_sum.append(val)

    obs_sum = np.hstack(obs_sum)
    labels = np.repeat(np.arange(n_models), n_samples)

    # Define the Random Forest classifier
    max_features = int(f_max_features * ref_table.shape[1])
    classifier = RandomForestClassifier(n_estimators = n_trees,
                                        max_features = max_features,
                                        bootstrap = True,
                                        random_state = random_seed)

    classifier.fit(ref_table, labels)
    best_model = int(classifier.predict([obs_sum]))

    # Compute missclassification error rate
    pred_prob = classifier.predict_proba(ref_table)
    pred_error = 1 - np.take(pred_prob.T, labels)

    # Estimate a regression function with prediction error as response 
    # on summary statitistics of the reference table
    regressor = RandomForestRegressor(n_estimators = n_trees)
    regressor.fit(ref_table, pred_error)
    prob_best_model = 1 - regressor.predict([obs_sum])

    return root_statistics[best_model], prob_best_model.item()
    