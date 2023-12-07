#!/usr/bin/python

##########################################################################################################
### Imports
##########################################################################################################

# External
import argparse
import numpy as np
import optuna
import os
import pandas as pd

from joblib import dump as optimiser_dump, load as optimiser_load
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler, MOTPESampler
from typing import Any, Callable, Tuple

# Internal
from root_system_lib.config import add_argument, Config, construct_interval
from root_system_lib.constants import APEX_DIAM, ORGAN_ORIGIN, SCALE_FACTOR, ROOT_TYPES
from root_system_lib.distances import ROOT_DISTANCES
from root_system_lib.parameters import DeVriesParameters
from root_system_lib.random_generation import get_rng
from root_system_lib.root import RootNodeMap
from root_system_lib.stats import exec_root_stats_map, read_stats_data, get_root_stats_map,calculate_objectives, distance_fun,read_simulated_stats_file


##########################################################################################################
### Parameters
##########################################################################################################

ROOT_STATS_MAP, _ = get_root_stats_map()

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Optimise a synthetic root system.'
    )

    # Optuna
    add_argument(parser, "--experiment_name", "root_gen_optimise", "The optimisation experiment name", str)
    add_argument(parser, "--sampler", "tpes", "The optimisation sampling algorithm", str, choices = ["tpes", "cmaes", "nsga", "motpes"])
    add_argument(parser, "--n_trials", 1000, "The number of optimisation trials to perform")
    add_argument(parser, "--n_jobs", -1, "The number of trials to run in parallel")
    add_argument(parser, "--gc_after_trial", 0, "Perform garbage collection after each trial", choices = [0, 1])

    add_argument(parser, "--distance", "euclidean", "The data dissimilarity metric", str, choices = ["euclidean"])
    add_argument(parser, "--load_optimiser", 0, "Load existing optimiser results", choices = [0, 1])
    add_argument(parser, "--db_name", "/root_gen_optimise.db", "The trial database name", str)
    add_argument(parser, "--db_engine", "sqlite", "The trial database engine", str, choices = ["sqlite", "mysql", "postgres"])
    add_argument(parser, "--use_db", 0, "Write trial results to a database", choices = [0, 1])

    # System
    add_argument(parser, "--from_config", 0, "Override parameters using a root_config.yaml file", choices = [0, 1])
    add_argument(parser, "--min_vattempts", 10, "The minimum number of attempts to fix each validation error")
    add_argument(parser, "--max_vattempts", 15, "The maximum number of attempts to fix each validation error")
    add_argument(parser, "--min_correction_angle", 75, "The minimum rotation angle about the y-axis for out-of-bounds roots")
    add_argument(parser, "--max_correction_angle", 105, "The maximum rotation angle about the y-axis for out-of-bounds roots")
    add_argument(parser, "--visualise", 0, "Visualise the trial results", choices = [0, 1])

    # de Vries et al. (2021) Mycorrhizal associations change root functionality...
    add_argument(parser, "--species", 0, "The species associated with the root system", choices = [0, 1])

    # Input 
    add_argument(parser, "--obs_file", "data/root_obs.csv", "The observed root data file name", str)
    add_argument(parser, "--stats_file", "data/summary_stats/root_stats.csv", "The observed root statistics file name", str)
    # add_argument(parser, "--stats_file", "data/root_stats.csv", "The observed root statistics file name", str)
    add_argument(parser, "--calc_statistics", 0, "Calculate summary statistics from the observed root data", choices = [0, 1])

    # Ouput
    add_argument(parser, "--dir", "data/optimise", "Output directory", str)
    add_argument(parser, "--out", "root_optimise.csv", "Output CSV file name", str)
    add_argument(parser, "--round", 10, "Number of decimal places to round values when writing to file")
    add_argument(parser, "--bins", 10, "Number of bins when binning root data")

    # Roots
    add_argument(parser, "--root_stats", "rld_for_locations", "A comma-delimited list of simulated and real root statistics to compare", str)
    add_argument(parser, "--col_stats_map", None, "A comma-delimited list of mapped columns and statistics", str)
    add_argument(parser, "--min_order", 3, "The minimum root organ order")
    add_argument(parser, "--max_order", 6, "The maximum root organ order")
    add_argument(parser, "--origin_min", 1e-2, "The minimum distance of the initial primary root from the origin (cm)", float)
    add_argument(parser, "--origin_max", 1e-1, "The maximum distance of the initial primary root from the origin (cm)", float)
    add_argument(parser, "--r_ratio", 0.5, "Ratio of fine roots to structural roots based upon overall root diameter", float)
    add_argument(parser, "--froot_threshold", 2, "Threshold for classifying a root as a fine root, rather than a structural root (mm)", float)
    add_argument(parser, "--root_type", 1, "The root type to calculate summary statistics for", type=str, choices=ROOT_TYPES)

    ## Primary
    add_argument(parser, "--min_rnum_out", 8, "The minimum number of outer primary roots to be generated")
    add_argument(parser, "--max_rnum_out", 12, "The maximum number of outer primary roots to be generated")
    add_argument(parser, "--min_rnum_in", 6, "The minimum number of inner primary roots to be generated")
    add_argument(parser, "--max_rnum_in", 10, "The maximum number of inner primary roots to be generated")

    ### Size
    add_argument(parser, "--min_prlength", 20, "The minimum length of each primary root (cm)", float)
    add_argument(parser, "--max_prlength", 40, "The maximum length of each primary root (cm)", float)
    add_argument(parser, "--prlength_var", 3, "The variance for the interval of the length of each primary root (cm)", float)

    ## Secondary
    add_argument(parser, "--srnum_min", 2, "The minimum number of secondary roots to be generated")
    add_argument(parser, "--srnum_max", 10, "The maximum number of secondary roots to be generated")
    add_argument(parser, "--srnum_var", 1, "The variance for the interval of the number of secondary roots to be generated")
    add_argument(parser, "--min_snum_growth", 0.4, "The minimum growth rate for the number of secondary roots per root order")
    add_argument(parser, "--max_snum_growth", 0.6, "The maximum growth rate for the number of secondary roots per root order")

    ### Size
    add_argument(parser, "--min_srlength", 100, "The minimum length of each secondary root (cm)", float)
    add_argument(parser, "--max_srlength", 200, "The maximum length of each secondary root (cm)", float)
    add_argument(parser, "--srlength_var", 30, "The variance for the interval of the length of each secondary root (cm)", float)

    # Segments
    add_argument(parser, "--min_snum", 10, "The minimum number of segments per root")
    add_argument(parser, "--max_snum", 30, "The maximum number of segments per root")
    add_argument(parser, "--fix_seg", 0, "Use a fixed segment size for each root", int, choices = [0, 1])
    add_argument(parser, "--min_length_reduction", 0.4, "The minimum root length reduction factor", float)
    add_argument(parser, "--max_length_reduction", 0.6, "The maximum root length reduction factor", float)

    # Movement
    add_argument(parser, "--min_vary", 20, "Minimum random variation in degrees of subsequent segments along x, y, and z axes") 
    add_argument(parser, "--max_vary", 40, "Maximum random variation in degrees of subsequent segments along x, y, and z axes") 

    # Soil
    add_argument(parser, "--sblock_size", 1, "The (voxel) size of each soil block in cubic-cm", float)

    # Random 
    add_argument(parser, "--random_seed", None, "The random seed")

    return parser

if __name__ == "__main__":
    parser = get_parser()

##########################################################################################################
### Constants
##########################################################################################################

    # Config 
    CONFIG = Config().from_parser(parser)
    if CONFIG.get_as("from_config", bool):
        CONFIG.from_yaml(f"{CONFIG.get('dir')}/root_config.yaml")
    VAL_ATTEMPT_INTERVAL = construct_interval(CONFIG, "min_vattempts", "max_vattempts")  
    VALIDATION_PITCH_INTERVAL = construct_interval(CONFIG, "min_correction_angle", "max_correction_angle")  

    # Optuna
    STUDY_NAME = CONFIG.get("experiment_name")
    N_TRIALS = CONFIG.get("n_trials")
    N_JOBS = CONFIG.get("n_jobs")
    GC_AFTER_TRIAL = CONFIG.get("gc_after_trial")
    SAMPLER_KEY = CONFIG.get("sampler")
    OPTUNA_SAMPLERS = {
        "tpes" : TPESampler, 
        "cmaes" : CmaEsSampler, 
        "nsga" : NSGAIISampler, 
        "motpes" : MOTPESampler
    }
    DISTANCE_TYPE = CONFIG.get("distance")
    OPTIMISER_FILE = f"{CONFIG.get('dir')}/root_optimiser.pkl"
    LOAD_OPTIMISER = CONFIG.get_as("load_optimiser", bool)
    USE_DB = CONFIG.get_as("use_db", bool)
    if USE_DB:
        STORAGE = f"{CONFIG.get('db_engine')}://{CONFIG.get('db_name')}"
    else:
        STORAGE = None

    # Random 
    SEED = CONFIG.get("random_seed")
    RNG = get_rng(SEED)

    # Root
    PLANT = 1
    ROOT_PARAMS = DeVriesParameters(CONFIG.get("species"), RNG)
    ROOT_STATS = CONFIG.split("root_stats", ",")
    COL_STATS_MAP = CONFIG.get("col_stats_map")
    R_RATIO = CONFIG.get("r_ratio")
    FROOT_THRESHOLD = CONFIG.get("froot_threshold")
    ROOT_TYPE = CONFIG.get("root_type")

    ORDER_INTERVAL = construct_interval(CONFIG, "min_order", "max_order")  
    NO_ROOT_ZONE = ROOT_PARAMS.rzone
    DIAMETER_REDUCTION = ROOT_PARAMS.get_rdm()

    ## Primary
    OUTER_RNUM_INTERVAL = construct_interval(CONFIG, "min_rnum_out", "max_rnum_out")  
    INNER_RNUM_INTERVAL = construct_interval(CONFIG, "min_rnum_in", "max_rnum_in")  
    PRLENGTH_INTERVAL = construct_interval(CONFIG, "min_prlength", "max_prlength")  
    PRLENGTH_VARIANCE = CONFIG.get("prlength_var")

    ## Secondary
    SRNUM_INTERVAL = construct_interval(CONFIG, "srnum_min", "srnum_max")  
    SRNUM_VARIANCE = CONFIG.get("srnum_var")
    SR_GROWTH_INTERVAL = construct_interval(CONFIG, "min_snum_growth", "max_snum_growth")  
    SRLENGTH_INTERVAL = construct_interval(CONFIG, "min_srlength", "max_srlength")  
    SRLENGTH_VARIANCE = CONFIG.get("srlength_var")

    # Segments
    SNUM_INTERVAL = construct_interval(CONFIG, "min_snum", "max_snum")  
    FIXED_SEG_LENGTH = CONFIG.get_as("fix_seg", bool)
    LENGTH_REDUCTION_INTERVAL = construct_interval(CONFIG, "min_length_reduction", "max_length_reduction")  
    
    # Movement
    VARIANCE_INTERVAL = construct_interval(CONFIG, "min_vary", "max_vary")  

    # Input 
    OBS_FILE = CONFIG.get("obs_file")
    STATS_FILE = CONFIG.get("stats_file")

    # Ouput
    DIR = CONFIG.get("dir")
    OUT = f"{DIR}/{CONFIG.get('out')}"
    ROUND = CONFIG.get("round")
    BINS = CONFIG.get("bins")

    # Location
    # Set coordinates of root system
    # The base node is located at origin (0, 0, 0)
    # We include a small deviation from this origin for all child nodes to prevent their dimensions from being exactly 0
    ORIGIN_NOISE_RANGE = CONFIG.get("origin_min"), CONFIG.get("origin_max")

    # Soil
    SOIL_BLOCK_SIZE = CONFIG.get("sblock_size")

    # Args
    KWARGS_MAP = {
        "bins": BINS,
        "depth_cum_col": "z",
        "horizontal_cum_cols": ["x", "y"],
        "rtd": ROOT_PARAMS.rtd,
        "soil_block_size": SOIL_BLOCK_SIZE,
        "as_scalar": True,
        "values_only": False,
        "x_locations": [0.1],  # Example x coordinates
        "y_locations": [1.5],  # Example y coordinates
        "x_tolerance": 0.2,  # Example tolerance for x
        "depth_interval": 0.1,  # Example depth interval in meters
        "ROOT_GROUP": ""
    }


##########################################################################################################
### Main
##########################################################################################################

def objective(trial: optuna.trial.Trial, compute_distance: Callable, root_stats_map: dict, kwargs_map: dict, 
    obs_statistics: dict, e: float):
    """Objective function for root simulation."""
    # Sample parameter values
    max_order = trial.suggest_int("max_order", *ORDER_INTERVAL) 
    num_segs = trial.suggest_int("num_segs", *SNUM_INTERVAL)
    length_reduction = trial.suggest_float("length_reduction", *LENGTH_REDUCTION_INTERVAL)
    snum_growth = trial.suggest_float("snum_growth", *SR_GROWTH_INTERVAL) 
    vary = trial.suggest_int("vary", *VARIANCE_INTERVAL)
    max_attempts = trial.suggest_int("max_attempts", *VAL_ATTEMPT_INTERVAL)
    validation_pitch = trial.suggest_int("validation_pitch", *VALIDATION_PITCH_INTERVAL)
    proot_num = trial.suggest_int("outer_rnum_int", *OUTER_RNUM_INTERVAL), trial.suggest_int("inner_rnum_int", *INNER_RNUM_INTERVAL)

    def __add_interval(interval: Tuple[Any], name1: str, name2: str, suggestion, variance: Any):
        x1, x2 = interval
        x1 = suggestion(name1, x1 - variance, x1 + variance)
        x2 = suggestion(name2, x2 - variance, x2 + variance)
        return np.array([x1, x2]) 

    prlength_range = __add_interval(PRLENGTH_INTERVAL, "min_prlength", "max_prlength", trial.suggest_float, PRLENGTH_VARIANCE)
    srlength_range = __add_interval(SRLENGTH_INTERVAL, "min_srlength", "max_srlength", trial.suggest_float, SRLENGTH_VARIANCE)
    sroot_num = __add_interval(SRNUM_INTERVAL, "srnum_min", "srnum_max", trial.suggest_int, SRNUM_VARIANCE)

    # Create synthetic root data
    base_diameter = ROOT_PARAMS.get_dinit() * max_order
    root_map = RootNodeMap(max_order, ORGAN_ORIGIN, num_segs, prlength_range, FIXED_SEG_LENGTH, 
        length_reduction, base_diameter, DIAMETER_REDUCTION, APEX_DIAM, SCALE_FACTOR, RNG)
    
    root_map.init_plant(PLANT)
    root_map.construct_roots(vary, proot_num, sroot_num, snum_growth, srlength_range, FROOT_THRESHOLD, R_RATIO)
    root_map.position_secondary_roots()
    root_map.position_primary_roots(proot_num, ORIGIN_NOISE_RANGE)
    root_map.validate(NO_ROOT_ZONE, validation_pitch, max_attempts)
    sim_df = root_map.to_dataframe(ROUND)

    sim_statistics = exec_root_stats_map(sim_df, root_stats_map, ROOT_STATS, kwargs_map)

    root_distance = calculate_objectives(obs_statistics, sim_statistics, ROOT_STATS, distance_fun)
    
    
    return root_distance
    


def get_sampler(samplers: dict, sampler_key: str, seed: int):
    """Get Optuna sampler."""
    # Define constructor kwargs
    tpes = {
        "consider_prior": True,
        "prior_weight": 1.0,
        "consider_endpoints": False,
        "n_startup_trials": 10,
        #"multivariate": True,
        "seed": seed,
        #"constant_liar": True
    }

    cmaes = {
        "n_startup_trials": 1,
        "warn_independent_sampling": True,
        "consider_pruned_trials": True,
        "restart_strategy": "ipop",
        "seed": seed
    }

    nsga = {
        "population_size": 50,
        "crossover_prob": 0.9,
        "swapping_prob": 0.5,
        "seed": seed
    }

    motpes = {
        "consider_prior": True,
        "prior_weight": 1.0,
        "consider_endpoints": True,
        "n_startup_trials": 10,
        "seed": seed
    }

    sampler_kwargs = {
        "tpes" : tpes, 
        "cmaes" : cmaes, 
        "nsga" : nsga, 
        "motpes" : motpes
    }

    # Instantiate sampler with kwargs
    kwargs = sampler_kwargs[sampler_key]
    sampler = samplers.get(sampler_key)(**kwargs)
    return sampler

def main() -> None:
    if "rld_for_locations" in ROOT_STATS:
        # If ROOT_STATS is 'rld_for_locations', directly read the simulated statistics file
        print("Reading simulated stats from:", STATS_FILE, ROOT_STATS)
        df = pd.read_csv(STATS_FILE)
        print(df)  # Print the DataFrame
        obs_statistics = read_simulated_stats_file(STATS_FILE)
    else:
        # Otherwise, use the read_stats_data function to process other types of statistics
        obs_statistics, _ = read_stats_data(CONFIG.get_as("calc_statistics", bool), 
                                            OBS_FILE, STATS_FILE, ROOT_STATS, KWARGS_MAP, 
                                            COL_STATS_MAP, ROOT_TYPE)

    if not obs_statistics.empty:
        directions = ["minimize"]
    else:
        directions = None


    sampler = get_sampler(OPTUNA_SAMPLERS, SAMPLER_KEY, SEED)
    if LOAD_OPTIMISER:
        study = optimiser_load(OPTIMISER_FILE)
    else:
        study = optuna.create_study(storage = STORAGE, sampler = sampler, study_name = STUDY_NAME, 
            load_if_exists = USE_DB, directions = directions)

    compute_distance = ROOT_DISTANCES.get(DISTANCE_TYPE)
    e = 1
    study.optimize(lambda trial: objective(trial, compute_distance, ROOT_STATS_MAP, KWARGS_MAP, obs_statistics, e), 
        n_trials = N_TRIALS, n_jobs = N_JOBS, gc_after_trial = GC_AFTER_TRIAL, show_progress_bar = True)

    study.trials_dataframe().to_csv(OUT, index = False)
    print(f"Trial results written to {OUT}")
    config_yaml: str = f"{DIR}/root_config.yaml"
    CONFIG.to_yaml(config_yaml)
    print(f"Configuration written to {config_yaml}")

    optimiser_file = f"{DIR}/root_optimiser.pkl"
    optimiser_dump(study, optimiser_file)
    print(f"Optimiser written to {optimiser_file}")

    # if CONFIG.get_as("visualise", bool):

    #     def __plot_results(plot_func: Callable, i: int, obs_statistic: str, plot_name: str):
    #         plot_func(study, target = lambda t: t.values[i], 
    #             target_name = obs_statistic).write_image(f"{DIR}/{obs_statistic}_{plot_name}.png")

        # for i, obs_statistic in enumerate(obs_statistics):
        #     __plot_results(optuna.visualization.plot_contour, i, obs_statistic, "contour")
        #     __plot_results(optuna.visualization.plot_edf, i, obs_statistic, "edf")
        #     __plot_results(optuna.visualization.plot_optimization_history, i, obs_statistic, "optimization_history")
        #     __plot_results(optuna.visualization.plot_parallel_coordinate, i, obs_statistic, "parallel_coordinate")
        #     __plot_results(optuna.visualization.plot_param_importances, i, obs_statistic, "param_importances")
        #     __plot_results(optuna.visualization.plot_slice, i, obs_statistic, "slice")

if __name__ == "__main__":
    main()
