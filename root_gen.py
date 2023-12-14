#!/usr/bin/python

##########################################################################################################
### Imports
##########################################################################################################

# External
import argparse
import numpy as np
import os

# Internal
from root_system_lib.config import add_argument, Config
from root_system_lib.constants import APEX_DIAM, ORGAN_ORIGIN, SCALE_FACTOR
from root_system_lib.parameters import DeVriesParameters
from root_system_lib.random_generation import get_rng
from root_system_lib.root import RootNodeMap
from root_system_lib.stats import calc_root_stats

##########################################################################################################
### Parameters
##########################################################################################################

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Generate a synthetic root system.'
    )

    # System
    add_argument(parser, "--from_config", 0, "Override parameters using a root_config.yaml file", choices = [0, 1])
    add_argument(parser, "--max_vattempts", 15, "The maximum number of attempts to fix each validation error")
    add_argument(parser, "--correction_angle", 90, "The rotation angle about the y-axis for out-of-bounds roots")
    add_argument(parser, "--visualise", 0, "Visualise the root generation process at each transformation step", choices = [0, 1])

    # de Vries et al. (2021) Mycorrhizal associations change root functionality...
    add_argument(parser, "--species", 0, "The species associated with the root system", choices = [0, 1])

    # Ouput
    add_argument(parser, "--dir", "data/root_sim", "Output directory", str)
    add_argument(parser, "--out", "root_sim.csv", "Output CSV file name", str)
    add_argument(parser, "--round", 10, "Number of decimal places to round values when writing to file")
    add_argument(parser, "--stats", 1, "Calculate statistics for the root system", choices = [0, 1])
    add_argument(parser, "--bins", 10, "Number of bins when binning root data")

    # Roots
    add_argument(parser, "--nplants", 1, "The number of plants to generate roots for")
    add_argument(parser, "--morder", 4, "The maximum root organ order")
    add_argument(parser, "--r_ratio", 0.5, "Ratio of fine roots to structural roots based upon overall root diameter", float)
    add_argument(parser, "--froot_threshold", 1.5, "Threshold for classifying a root as a fine root, rather than a structural root (mm)", float)

    ## Primary
    add_argument(parser, "--rnum_out", 10, "The number of outer primary roots to be generated")
    add_argument(parser, "--rnum_in", 8, "The number of inner primary roots to be generated")
    ### Size
    add_argument(parser, "--min_prlength", 20, "The minimum length of each primary root (cm)", float)
    add_argument(parser, "--max_prlength", 30, "The maximum length of each primary root (cm)", float)
    # add_argument(parser, "--diam", 0.012, "The base diameter of the first segment of each primary root (cm)", float) # Use param
    add_argument(parser, "--origin_min", 1e-2, "The minimum distance of the initial primary root from the origin (cm)", float)
    add_argument(parser, "--origin_max", 1e-1, "The maximum distance of the initial primary root from the origin (cm)", float)

    ## Secondary
    add_argument(parser, "--srnum_min", 3, "The minimum number of secondary roots to be generated")
    add_argument(parser, "--srnum_max", 4, "The maximum number of secondary roots to be generated")
    add_argument(parser, "--snum_growth", 0.5, "The growth rate for the number of secondary roots per root order")

    ### Size
    add_argument(parser, "--min_srlength", 100, "The minimum length of each secondary root (cm)", float)
    add_argument(parser, "--max_srlength", 220, "The maximum length of each secondary root (cm)", float)

    # Segments
    add_argument(parser, "--snum", 10, "The number of segments per root")
    add_argument(parser, "--fix_seg", 0, "Use a fixed segment size for each root", int, choices = [0, 1])
    add_argument(parser, "--length_reduction", 0.5, "The root length reduction factor", float)

    # Movement
    add_argument(parser, "--vary", 30, "Random variation in degrees of subsequent segments along x, y, and z axes") 

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
    MAX_ATTEMPTS = CONFIG.get("max_vattempts")
    VALIDATION_PITCH = CONFIG.get("correction_angle")
    VISUALISE_ROOTS = CONFIG.get_as("visualise", bool)

    # Random 
    RNG = get_rng(CONFIG.get("random_seed"))

    # Root
    ROOT_PARAMS = DeVriesParameters(CONFIG.get("species"), RNG)
    N_PLANTS = CONFIG.get("nplants")
    MAX_ORDER = CONFIG.get("morder")
    NUM_SEGS = CONFIG.get("snum")
    R_RATIO = CONFIG.get("r_ratio")
    FROOT_THRESHOLD = CONFIG.get("froot_threshold")

    FIXED_SEG_LENGTH = CONFIG.get_as("fix_seg", bool)
    LENGTH_REDUCTION = CONFIG.get("length_reduction")
    VARY = CONFIG.get("vary")
    # Roots
    PROOT_NUM = CONFIG.get("rnum_out"), CONFIG.get("rnum_in")
    SROOT_NUM = CONFIG.get("srnum_min"), CONFIG.get("srnum_max")
    SNUM_GROWTH = CONFIG.get("snum_growth")
    NO_ROOT_ZONE = ROOT_PARAMS.rzone

    ### Size
    PRLENGTH_RANGE = np.array([CONFIG.get("min_prlength"), CONFIG.get("max_prlength")])
    SRLENGTH_RANGE = np.array([CONFIG.get("min_srlength"), CONFIG.get("max_srlength")])
    BASE_DIAMETER = ROOT_PARAMS.get_dinit() * MAX_ORDER
    DIAMETER_REDUCTION = ROOT_PARAMS.get_rdm()

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

    ROOT_MAP = RootNodeMap(MAX_ORDER, ORGAN_ORIGIN, NUM_SEGS, PRLENGTH_RANGE, FIXED_SEG_LENGTH, 
        LENGTH_REDUCTION, BASE_DIAMETER, DIAMETER_REDUCTION, APEX_DIAM, SCALE_FACTOR, RNG)
        
##########################################################################################################
### Main
##########################################################################################################

def main() -> None:
    for plant in range(1, N_PLANTS + 1):
        ROOT_MAP.init_plant(plant)
        ROOT_MAP.construct_roots(VARY, PROOT_NUM, SROOT_NUM, SNUM_GROWTH, SRLENGTH_RANGE, R_RATIO, VISUALISE_ROOTS)
        ROOT_MAP.position_secondary_roots(VISUALISE_ROOTS)
        ROOT_MAP.position_primary_roots(PROOT_NUM, ORIGIN_NOISE_RANGE, VISUALISE_ROOTS)
        ROOT_MAP.validate(NO_ROOT_ZONE, VALIDATION_PITCH, MAX_ATTEMPTS)
    df = ROOT_MAP.to_dataframe(ROUND)
    
    # Saving roots
    os.makedirs(DIR, exist_ok = True)
    df.to_csv(OUT, index = False)
    print(f"Generated roots written to {OUT}")
    # Saving config
    config_yaml: str = f"{DIR}/root_config.yaml"
    CONFIG.to_yaml(config_yaml)
    print(f"Configuration written to {config_yaml}")
    # Saving stats
    if CONFIG.get_as("stats", bool):
        calc_root_stats(df, f"{DIR}/root_stats.csv", BINS)

if __name__ == "__main__":
    main()