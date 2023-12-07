#!/usr/bin/python

##########################################################################################################
### Imports
##########################################################################################################

# External
import argparse
import pandas as pd
from os.path import join as path_join

# Internal
from root_system_lib.config import add_argument, Config
from root_system_lib.parameters import DeVriesParameters
from root_system_lib.random_generation import get_rng
from root_system_lib.stats import exec_root_stats_map, get_root_stats_map, calc_rld_and_sum_for_multiple_locations

##########################################################################################################
### Parameters
##########################################################################################################

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Generate summary statistics from observed data.'
    )

    # Data
    add_argument(parser, "--obs_file", "data/root_sim/root_sim.csv", "The observed root data file name", str)
    add_argument(parser, "--group_by", "", "A comma-delimited list of column names to group by", str)
    add_argument(parser, "--enable_group_by", 0, "Enable group by functionality", choices = [0, 1])
    add_argument(parser, "--copy_cols", 1, "Whether to directly copy columns from the existing dataframe", choices = [0, 1])
    add_argument(parser, "--col_list", "length", "A comma-delimited list of column names to group by", str)

    # Ouput
    add_argument(parser, "--dir", "data/summary_stats", "Output directory", str)
    add_argument(parser, "--out", "root_stats.csv", "Output CSV file name", str)
    add_argument(parser, "--round", 10, "Number of decimal places to round values when writing to file")
    add_argument(parser, "--bins", 10, "Number of bins when binning root data")
    add_argument(parser, "--by_group", 0, "Create on csv file per group", choices = [0, 1])

    # Roots
    add_argument(parser, "--root_stats", "rld_for_locations=depth_cum_col", "A comma-delimited list of summary statistics mapped to a column name", str)

    # Soil
    add_argument(parser, "--sblock_size", 0.3, "The (voxel) size of each soil block in cubic-cm", float)

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

    # Data
    OBS_FILE = CONFIG.get("obs_file")
    if OBS_FILE is None:
        raise Exception("Must supply an observed data file.")
    GROUP_BY = CONFIG.split("group_by", ",")
    if CONFIG.get_as("copy_cols", bool):
        COL_LIST = CONFIG.split("col_list", ",")
    else:
        COL_LIST = []

    # Ouput
    DIR = CONFIG.get("dir")
    OUT = f"{DIR}/{CONFIG.get('out')}"
    ROUND = CONFIG.get("round")
    BINS = CONFIG.get("bins")

    # Random 
    SEED = CONFIG.get("random_seed")
    RNG = get_rng(SEED)

    # Root
    ROOT_STATS = CONFIG.split("root_stats", ",")
    STATS_LIST = [pair.split('=')[0] for pair in ROOT_STATS]
    print(STATS_LIST)
    # Soil
    SOIL_BLOCK_SIZE = CONFIG.get("sblock_size")

    ROOT_PARAMS = DeVriesParameters(CONFIG.get("species"), RNG)

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

# def gen_summary_stats(df: pd.DataFrame, col_stats_pairs: list, root_stats_map: dict, col_list: list, kwargs_map: dict) -> pd.DataFrame:
#     root_stat_map = {}
#     for stat, col in col_stats_pairs:
#         kwargs_map[f"{stat}_col"] = col
#         root_stats = exec_root_stats_map(df, root_stats_map, [stat], kwargs_map)[stat]
#         print(f"Debug: root_stats for {stat} is of type {type(root_stats)} and has content {root_stats}")
#         # Check if the stat is for 'calc_rld_and_sum_for_multiple_locations'
#         if stat == 'rld_for_locations':
#             return root_stats  # Directly return the result for this specific stat

#         # Existing logic for other stats
#         for i in range(root_stats.shape[0]):
#             root_stat_map[f"{col}_{stat}_var_{i + 1}"] = pd.Series(root_stats[i])

#     for col in col_list:
#         df_value = df[col].unique() 
#         root_stat_map[col] = pd.Series(df_value)
   
#     root_stats_df = pd.DataFrame(root_stat_map)
#     return root_stats_df

def gen_summary_stats(df, stats_list, root_stats_map, kwargs_map):
    for stat in stats_list:
        if stat == 'rld_for_locations':
            # Special handling for 'rld_for_locations'
            # exec_root_stats_map returns the DataFrame directly for this stat
            return exec_root_stats_map(df, root_stats_map, [stat], kwargs_map)

    # Calculate other statistics and store results in a dictionary
    stats_results = {}
    for stat in stats_list:
        stats_results[stat] = exec_root_stats_map(df, root_stats_map, [stat], kwargs_map)[stat]

    return stats_results


def apply_gen_summary_stats(df, stats_list, root_stats_map, kwargs_map):
    return gen_summary_stats(df, STATS_LIST, root_stats_map, KWARGS_MAP)



def main() -> None:    
    df = pd.read_csv(OBS_FILE)
    root_stats_map, _ = get_root_stats_map()
    # col_stats_pairs = [ stat.split("=") for stat in ROOT_STATS ]
   
    if CONFIG.get_as("enable_group_by", bool) and len(GROUP_BY) > 0:
        df = df.groupby(GROUP_BY)
        # df = df.apply(gen_summary_stats, col_stats_pairs, root_stats_map, COL_LIST, KWARGS_MAP) 
        df = apply_gen_summary_stats(df, STATS_LIST, root_stats_map, KWARGS_MAP)
    else:
        # df = gen_summary_stats(df, col_stats_pairs, root_stats_map, COL_LIST, KWARGS_MAP)
        df = apply_gen_summary_stats(df, STATS_LIST, root_stats_map, KWARGS_MAP)

    # Convert 'depth_bin' column to string if it exists
    if 'depth_bin' in df.columns:
        df['depth_bin'] = df['depth_bin'].astype(str)

    config_yaml: str = path_join(DIR, "root_config.yaml")
    CONFIG.to_yaml(config_yaml)
    print(f"Configuration written to {config_yaml}")

    if CONFIG.get_as("by_group", bool) and CONFIG.get_as("enable_group_by", bool):
        df = df.groupby(GROUP_BY)
        for group in df:
            group_vals, df_g = group
            df_file_name = [ f"{GROUP_BY[i]}_{group_val}" for i, group_val in enumerate(group_vals) ]
            df_file_name = '_'.join(df_file_name)
            df_file_name = path_join(DIR, f"{df_file_name}_{CONFIG.get('out')}")
            df_g.reset_index().to_csv(df_file_name, index = False)
            print(f"Statistics written to {df_file_name}")
    else:
        df.reset_index().to_csv(OUT, index = False)
        print(f"Statistics written to {OUT}")

if __name__ == "__main__":
    main()
