#!/usr/bin/python

##########################################################################################################
### Imports
##########################################################################################################

# External
import argparse
import arviz as az
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import warnings
warnings.filterwarnings('ignore')
from typing import Any, Callable, Tuple

# Internal
from root_system_lib.abc import select_model
from root_system_lib.config import add_argument, Config, construct_interval
from root_system_lib.constants import APEX_DIAM, DISTRIBUTION_METHODS, ORGAN_ORIGIN, SCALE_FACTOR, ROOT_TYPES
from root_system_lib.distances import ROOT_DISTANCES
from root_system_lib.parameters import DeVriesParameters
from root_system_lib.random_generation import get_rng
from root_system_lib.root import RootNodeMap
from root_system_lib.stats import exec_root_stats_map, get_root_stats_map, read_stats_data

##########################################################################################################
### Parameters
##########################################################################################################

ROOT_STATS_MAP, available_stats = get_root_stats_map()

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Perform model selection on a synthetic root system using ABC random forest.'
    )

    # System
    add_argument(parser, "--from_config", 0, "Override parameters using a root_config.yaml file", choices = [0, 1])
    add_argument(parser, "--min_vattempts", 10, "The minimum number of attempts to fix each validation error")
    add_argument(parser, "--max_vattempts", 15, "The maximum number of attempts to fix each validation error")
    add_argument(parser, "--min_correction_angle", 75, "The minimum rotation angle about the y-axis for out-of-bounds roots")
    add_argument(parser, "--max_correction_angle", 105, "The maximum rotation angle about the y-axis for out-of-bounds roots")
    add_argument(parser, "--visualise", 0, "Visualise the trial results", choices = [0, 1])

    # ABC-SMC
    add_argument(parser,"--draws", 2, "The number of samples to draw from the posterior. And also the number of independent chains")
    add_argument(parser, "--steps", 2, "The number of steps for each Markov Chain")
    add_argument(parser, "--chains", 1, "The number of chains to sample. Running independent chains is important for some convergence statistics")
    add_argument(parser, "--parallel", 0, "Distribute computations across cores if the number of cores is larger than 1", choices = [0, 1])
    add_argument(parser, "--distance", "euclidean", "The data dissimilarity metric", str, choices = ["euclidean"])

    # de Vries et al. (2021) Mycorrhizal associations change root functionality...
    add_argument(parser, "--species", 0, "The species associated with the root system", choices = [0, 1])

    # Input 
    add_argument(parser, "--obs_file", "data/root_obs.csv", "The observed root data file name", str)
    add_argument(parser, "--stats_file", "data/root_stats.csv", "The observed root statistics file name", str)
    add_argument(parser, "--calc_statistics", 1, "Calculate summary statistics from the observed root data", choices = [0, 1])

    # Ouput
    add_argument(parser, "--dir", "data/abc_rf", "Output directory", str)
    add_argument(parser, "--out", "root_abc_rf.csv", "Output CSV file name", str)
    add_argument(parser, "--round", 10, "Number of decimal places to round values when writing to file")
    add_argument(parser, "--bins", 10, "Number of bins when binning root data")

    # Roots
    add_argument(parser, "--root_stats", "depth_cum,horizontal_cum,total_diameter,total_length", "A comma-delimited list of root statistics to compare simulated and real data", str)
    # add_argument(parser, "--col_stats_map", None, "A comma-delimited list of mapped columns and statistics", str)
    add_argument(parser, "--min_order", 3, "The minimum root organ order")
    add_argument(parser, "--max_order", 4, "The maximum root organ order")
    add_argument(parser, "--origin_min", 1e-2, "The minimum distance of the initial primary root from the origin (cm)", float)
    add_argument(parser, "--origin_max", 1e-1, "The maximum distance of the initial primary root from the origin (cm)", float)
    add_argument(parser, "--r_ratio", 0.5, "Ratio of fine roots to structural roots based upon overall root diameter", float)
    add_argument(parser, "--froot_threshold", 1.5, "Threshold for classifying a root as a fine root, rather than a structural root (mm)", float)
    add_argument(parser, "--root_type", None, "The root type to calculate summary statistics for", str, choices = ROOT_TYPES)
    add_argument(parser, "--dinit", 0.11, "Initial root diameter (cm)", float)
    add_argument(parser, "--min_ibd", 0.08, "The minimum interbranching distance as a factor", float)
    add_argument(parser, "--max_ibd", 0.12, "The maximum interbranching distance as a factor", float)
    add_argument(parser, "--distribution", DISTRIBUTION_METHODS[0], "The method for distributing roots within the soil", str, choices = DISTRIBUTION_METHODS)
    add_argument(parser, "--sproportions", None, "The sampling proportions for the roots", str)

    ## Primary
    add_argument(parser, "--min_rnum_out", 8, "The minimum number of outer primary roots to be generated")
    add_argument(parser, "--max_rnum_out", 10, "The maximum number of outer primary roots to be generated")
    add_argument(parser, "--min_rnum_in", 6, "The minimum number of inner primary roots to be generated")
    add_argument(parser, "--max_rnum_in", 8, "The maximum number of inner primary roots to be generated")

    ### Size
    add_argument(parser, "--min_prlength", 20, "The minimum length of each primary root (cm)", float)
    add_argument(parser, "--max_prlength", 30, "The maximum length of each primary root (cm)", float)
    add_argument(parser, "--prlength_var", 3, "The variance for the interval of the length of each primary root (cm)", float)

    ## Secondary
    add_argument(parser, "--srnum_min", 1, "The minimum number of secondary roots to be generated")
    add_argument(parser, "--srnum_max", 4, "The maximum number of secondary roots to be generated")
    add_argument(parser, "--srnum_var", 1, "The variance for the interval of the number of secondary roots to be generated")
    add_argument(parser, "--min_snum_growth", 0.4, "The minimum growth rate for the number of secondary roots per root order")
    add_argument(parser, "--max_snum_growth", 0.6, "The maximum growth rate for the number of secondary roots per root order")

    ### Size
    add_argument(parser, "--min_srlength", 100, "The minimum length of each secondary root (cm)", float)
    add_argument(parser, "--max_srlength", 200, "The maximum length of each secondary root (cm)", float)
    add_argument(parser, "--srlength_var", 30, "The variance for the interval of the length of each secondary root (cm)", float)

    # Segments
    add_argument(parser, "--fix_seg", 0, "Use a fixed segment size for each root", int, choices = [0, 1])
    add_argument(parser, "--min_length_reduction", 0.4, "The minimum root length reduction factor", float)
    add_argument(parser, "--max_length_reduction", 0.6, "The maximum root length reduction factor", float)

    # Movement
    add_argument(parser, "--min_vary", 20, "Minimum random variation in degrees of subsequent segments along x, y, and z axes") 
    add_argument(parser, "--max_vary", 40, "Maximum random variation in degrees of subsequent segments along x, y, and z axes") 

    # Soil
    add_argument(parser, "--sblock_size", 1, "The (voxel) size of each soil block in cubic-cm", float)

    # Random Forest
    add_argument(parser, "--p_draws", 5, "The number of posterior draws for the random forest evaluation.")
    add_argument(parser, "--n_trees", 5, "The size of the random forest ensemble.")
    add_argument(parser, "--m_features", 0.5, "The proportion of the maximum number of features for random sub-sampling.", float)

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

    # ABC-SMC
    NDRAWS = CONFIG.get("draws")
    NSTEPS = CONFIG.get("steps")
    NCHAINS = CONFIG.get("chains")
    PARALLEL = CONFIG.get_as("distance", bool)
    DISTANCE_TYPE = CONFIG.get("distance")

    # Random 
    SEED = CONFIG.get("random_seed")
    RNG = get_rng(SEED)

    # Root
    PLANT = 1
    ROOT_PARAMS = DeVriesParameters(CONFIG.get("species"), RNG)
    ROOT_STATS = CONFIG.split("root_stats", ",")
    ROOT_STAT = [ROOT_STATS[0]]

    # COL_STATS_MAP = CONFIG.get("col_stats_map")
    R_RATIO = CONFIG.get("r_ratio")
    FROOT_THRESHOLD = CONFIG.get("froot_threshold")
    ROOT_TYPE = CONFIG.get("root_type")
    IBD_INTERVAL = construct_interval(CONFIG, "min_ibd", "max_ibd")  
    DISTRIBUTION_METHOD = CONFIG.get("distribution")
    SAMPLING_PROPORTIONS = CONFIG.get("sproportions")
    if SAMPLING_PROPORTIONS:
        SAMPLING_PROPORTIONS = SAMPLING_PROPORTIONS.split(",")
        
    ORDER_INTERVAL = construct_interval(CONFIG, "min_order", "max_order")  
    NO_ROOT_ZONE = ROOT_PARAMS.rzone
    DIAMETER_REDUCTION = R_RATIO

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

    # Data
    obs_df = pd.read_csv(OBS_FILE) 
    DF_COLS = obs_df.columns

    # Random Forest
    POSTERIOR_DRAWS = CONFIG.get("p_draws")
    N_TREES = CONFIG.get("n_trees")
    MAX_FEATURES = CONFIG.get("m_features")

    # Args
    KWARGS_MAP = {
        "bins": BINS,
        "depth_cum_col": "z",
        "horizontal_cum_cols": ["x", "y"],
        "rtd": ROOT_PARAMS.rtd,
        "soil_block_size": SOIL_BLOCK_SIZE,
        "as_scalar": True,
        "values_only": True
    }

##########################################################################################################
### Main
##########################################################################################################

def root_sim(max_order: int, ibd: float, length_reduction: float, snum_growth: float, vary: int, max_attempts: int,
    validation_pitch: int, outer_rnum_int: int, inner_rnum_int:int, min_prlength: float, max_prlength: float, 
    min_srlength: float, max_srlength: float, srnum_min: int, srnum_max: int):

    proot_num = outer_rnum_int, inner_rnum_int
    prlength_range = np.array([min_prlength, max_prlength])
    srlength_range = np.array([min_srlength, max_srlength])
    sroot_num = srnum_min, srnum_max

    # Create synthetic root data
    base_diameter = CONFIG.get("dinit") * max_order
    root_map = RootNodeMap(max_order, ORGAN_ORIGIN, R_RATIO, ibd, prlength_range, FIXED_SEG_LENGTH, 
        length_reduction, base_diameter, DIAMETER_REDUCTION, APEX_DIAM, SCALE_FACTOR, DISTRIBUTION_METHOD, RNG)

    root_map.init_plant(PLANT)
    root_map.construct_roots(vary, proot_num, sroot_num, snum_growth, srlength_range, FROOT_THRESHOLD, R_RATIO)
    root_map.position_secondary_roots(proportions = SAMPLING_PROPORTIONS)
    root_map.position_primary_roots(proot_num, ORIGIN_NOISE_RANGE, proportions = SAMPLING_PROPORTIONS)
    root_map.validate(NO_ROOT_ZONE, validation_pitch, max_attempts)
    sim_df = root_map.to_dataframe(ROUND)
    return sim_df.values 

def compute_distance(e: float, obs: np.ndarray, sim: np.ndarray):
    sim_df = pd.DataFrame(sim, columns = DF_COLS)
    root_distance = ROOT_DISTANCES.get(DISTANCE_TYPE)
    sim_statistic = exec_root_stats_map(sim_df, ROOT_STATS_MAP, ROOT_STAT, KWARGS_MAP, ROOT_TYPE).get(*ROOT_STAT) 
    return root_distance(e, obs, sim_statistic)

def fit_model(obs_statistic: pd.DataFrame, e: float):
    # Specification for ABC-SMC model.
    with pm.Model() as root_model:
        # Parameter priors
        max_order = pm.DiscreteUniform("max_order", *ORDER_INTERVAL)
        ibd = pm.Uniform("ibd", *IBD_INTERVAL)
        length_reduction = pm.Uniform("length_reduction", *LENGTH_REDUCTION_INTERVAL)
        snum_growth = pm.Uniform("snum_growth", *SR_GROWTH_INTERVAL)
        vary = pm.DiscreteUniform("vary", *VARIANCE_INTERVAL)
        max_attempts = pm.DiscreteUniform("max_attempts", *VAL_ATTEMPT_INTERVAL)
        validation_pitch = pm.DiscreteUniform("validation_pitch", *VALIDATION_PITCH_INTERVAL)
        outer_rnum_int = pm.DiscreteUniform("outer_rnum_int", *OUTER_RNUM_INTERVAL)
        inner_rnum_int = pm.DiscreteUniform("inner_rnum_int", *INNER_RNUM_INTERVAL)

        def __add_interval(interval: Tuple[Any], name1: str, name2: str, dist, variance: Any):
            x1, x2 = interval
            x1 = dist(name1, lower = x1 - variance, upper = x1 + variance)
            x2 = dist(name2, lower = x2 - variance, upper = x2 + variance)
            return x1, x2

        min_prlength, max_prlength = __add_interval(PRLENGTH_INTERVAL, "min_prlength", "max_prlength", pm.Uniform, PRLENGTH_VARIANCE)
        min_srlength, max_srlength = __add_interval(SRLENGTH_INTERVAL, "min_srlength", "max_srlength", pm.Uniform, SRLENGTH_VARIANCE)
        srnum_min, srnum_max = __add_interval(SRNUM_INTERVAL, "srnum_min", "srnum_max", pm.DiscreteUniform, SRNUM_VARIANCE)

        # Create simulator
        S = pm.Simulator(
            "S", 
            function = root_sim, 
            params = (max_order, ibd, length_reduction, snum_growth, vary, max_attempts, validation_pitch, outer_rnum_int, 
                inner_rnum_int, min_prlength, max_prlength, min_srlength, max_srlength, srnum_min, srnum_max),
                sum_stat = "identity",
                epsilon = e,
                observed = obs_statistic,
                distance = compute_distance
            )
    
    with root_model:
        # Draw from the posterior.
        trace, sim_data = pm.sample_smc(model = root_model, draws = NDRAWS, n_steps = NSTEPS, tune_steps = True, 
            kernel='ABC', save_sim_data = True, chains = NCHAINS, parallel = PARALLEL)

        return root_model, trace

def main() -> None:
    obs_statistics, _ = read_stats_data(CONFIG.get_as("calc_statistics", bool), OBS_FILE, STATS_FILE, 
        ROOT_STATS, KWARGS_MAP, ROOT_TYPE) 
    e = 1

    models = {}
    for root_stat in ROOT_STATS:
        print(f"\nFitting model for root statistic: {root_stat}")
        ROOT_STAT = [root_stat]
        obs_statistic = obs_statistics.get(root_stat) 
        model_results = fit_model(obs_statistic, e)
        models[root_stat] = model_results
    
    print(f"\nPerforming model selection")
    root_statistic, prob_best_model = select_model(models, obs_statistics, ROOT_STATS_MAP, ROOT_STATS, 
        KWARGS_MAP, DF_COLS, ROOT_TYPE, POSTERIOR_DRAWS, n_trees = N_TREES, f_max_features = MAX_FEATURES, random_seed = SEED)

    rf_statistic_df = pd.DataFrame({
        "root_statistic": root_statistic, 
        "prob_best_model": prob_best_model, 
        "n_trees": N_TREES,
        "n_samples": POSTERIOR_DRAWS,
        "max_features": MAX_FEATURES
    }, index = [0])

    rf_statistic_file = os.path.join(DIR, "rf_statistics.csv")
    rf_statistic_df.to_csv(rf_statistic_file, index = False)
    print(f"Random forest statistics written to {rf_statistic_file}")

    config_yaml: str = os.path.join(DIR, "root_config.yaml") 
    CONFIG.to_yaml(config_yaml)
    print(f"Configuration written to {config_yaml}")

if __name__ == "__main__":
    main()

