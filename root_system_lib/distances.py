"""
Root System Library

A library for computing dissimilarities between data from synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import numpy as np
import pandas as pd
from typing import Tuple

##########################################################################################################
### Library
##########################################################################################################

def distance_euclidean(e: float, obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes Euclidean distance between data.

    Parameters
    --------------
    e: float
        Epsilon parameter for compatability with PyMC3. 
    obs: array (float)  
        Observed data.
    sim: array (float)
        Simulated data.

    Returns
    ---------
    distance : float | array (float)
        The computed euclidean distances.
    """
    dist = np.linalg.norm(obs - sim)
    return dist * e




# Define your distance computation function (e.g., RMSE, MAE, etc.)
def distance_fun(observed_values, simulated_values):
    # Example with Mean Absolute Error (MAE)
    return np.mean(np.abs(observed_values - simulated_values))

def calculate_objectives(obs_statistics_df, sim_statistics_df, 
                        root_stats="rld_for_locations", compute_distance_func=distance_fun):
    objectives = []

    # Check if the specific statistic 'rld_for_locations' is the one we're interested in
    if 'rld_for_locations' in root_stats:
        # If yes, ensure that 'depth_bin' is a column in both observed and simulated DataFrames
        if 'depth_bin' in obs_statistics_df.columns and 'depth_bin' in sim_statistics_df.columns:
            # Calculate the objective for this statistic
            objective = calculate_objective(obs_statistics_df, sim_statistics_df, compute_distance_func)
            objectives.append(objective)
    return objectives


def calculate_objective(obs_statistics_df, sim_statistics_df, 
                        compute_distance_func=distance_fun, include_root_type=False):
    # Assuming 'depth_bin' is in the DataFrame, cast it to string type
    obs_statistics_df['depth_bin'] = obs_statistics_df['depth_bin'].astype(str)
    sim_statistics_df['depth_bin'] = sim_statistics_df['depth_bin'].astype(str)

    # Group by 'depth_bin' and optionally by 'root_type'
    if include_root_type and 'root_type' in obs_statistics_df.columns:
        grouped_obs = obs_statistics_df.groupby(['depth_bin', 'root_type'])
        grouped_sim = sim_statistics_df.groupby(['depth_bin', 'root_type'])
    else:
        grouped_obs = obs_statistics_df.groupby(['depth_bin'])
        grouped_sim = sim_statistics_df.groupby(['depth_bin'])
    
    # Initialize an empty list to hold the objective scores
    objective_scores = []

    # Iterate over unique combinations of 'depth_bin' and optionally 'root_type'
    for group_keys, obs_group in grouped_obs:
        if group_keys in grouped_sim.groups:
            sim_group = grouped_sim.get_group(group_keys)
            # Calculate the objective score using a custom distance function
            distance = compute_distance_func(obs_group['rld'], sim_group['rld'])
            objective_scores.append(distance)

    # Sum or average the objective scores as needed
    total_objective = sum(objective_scores)
    return total_objective

def calculate_objectives_with_numpy(e: float, observed_values, sim_values) -> float:
    # Check if both observed_values and sim_values are NumPy arrays
    if not isinstance(observed_values, np.ndarray) or not isinstance(sim_values, np.ndarray):
        raise ValueError("Both observed_values and sim_values must be NumPy arrays")

    # Extract the unique depth bin indices from both arrays
    obs_depth_bins = np.unique(observed_values[:, 1])  # Assuming depth_bin_idx is in column 1
    sim_depth_bins = np.unique(sim_values[:, 1])
    all_depth_bins = np.unique(np.concatenate([obs_depth_bins, sim_depth_bins]))

    # Initialize the total distance
    total_distance = 0

    # Iterate through each depth bin index
    for bin_idx in all_depth_bins:
        obs_bin_mask = observed_values[:, 1] == bin_idx
        sim_bin_mask = sim_values[:, 1] == bin_idx

        obs_group = observed_values[obs_bin_mask]
        sim_group = sim_values[sim_bin_mask]

        # Check if both groups have data
        if obs_group.size > 0 and sim_group.size > 0:
            # Assuming 'rld' is in column 3
            obs_rld = obs_group[:, 3].astype(float)
            sim_rld = sim_group[:, 3].astype(float)

            # Calculate the Euclidean distance for this group
            dist = np.linalg.norm(obs_rld - sim_rld)
            total_distance += dist

    # Return the scaled total distance
    print("Total distance:", total_distance)
    return total_distance * e

ROOT_DISTANCES = {
    "euclidean": distance_euclidean,
    "customized": calculate_objectives_with_numpy
}
 