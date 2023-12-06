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
from root_system_lib.stats import exec_root_stats_map, get_root_stats_map

##########################################################################################################
### Parameters
##########################################################################################################

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Generate summary statistics from observed data.'
    )

    # Data
    add_argument(parser, "--obs_file", "data/field/apple_tree_root_data.csv", "The observed root data file name", str)
    add_argument(parser, "--group_by", "experiment_id,tree_id,age", "A comma-delimited list of column names to group by", str)
    add_argument(parser, "--enable_group_by", 1, "Enable group by functionality", choices = [0, 1])
    add_argument(parser, "--copy_cols", 1, "Whether to directly copy columns from the existing dataframe", choices = [0, 1])
    add_argument(parser, "--col_list", "mean_rld_cm_cm3,total_root_to_1m__kg_per_plant", "A comma-delimited list of column names to group by", str)

    # Ouput
    add_argument(parser, "--dir", "data/summary_stats", "Output directory", str)
    add_argument(parser, "--out", "root_stats.csv", "Output CSV file name", str)
    add_argument(parser, "--round", 10, "Number of decimal places to round values when writing to file")
    add_argument(parser, "--bins", 10, "Number of bins when binning root data")
    add_argument(parser, "--by_group", 1, "Create on csv file per group", choices = [0, 1])

    # Roots
    add_argument(parser, "--root_stats", "average_rld_depth=soil_depth_m,depth_cum=soil_depth_m,depth_cum=radial_distance_from_stem_m", "A comma-delimited list of summary statistics mapped to a column name", str)

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
        "values_only": False
    }

##########################################################################################################
### Main
##########################################################################################################

def gen_summary_stats(df: pd.DataFrame, col_stats_pairs: list, root_stats_map: dict, col_list: list, kwargs_map: dict) -> pd.DataFrame:
    root_stat_map = {}
    for stat, col in col_stats_pairs:
        kwargs_map[f"{stat}_col"] = col
        root_stats = exec_root_stats_map(df, root_stats_map, [stat], kwargs_map)[stat]
        for i in range(root_stats.shape[0]):
            root_stat_map[f"{col}_{stat}_var_{i + 1}"] = pd.Series(root_stats[i])

    for col in col_list:
        df_value = df[col].unique() 
        root_stat_map[col] = pd.Series(df_value)
   
    root_stats_df = pd.DataFrame(root_stat_map)
    return root_stats_df

def main() -> None:    
    df = pd.read_csv(OBS_FILE)
    root_stats_map, _ = get_root_stats_map()
    col_stats_pairs = [ stat.split("=") for stat in ROOT_STATS ]

    if CONFIG.get_as("enable_group_by", bool) and len(GROUP_BY) > 0:
        df = df.groupby(GROUP_BY)
        df = df.apply(gen_summary_stats, col_stats_pairs, root_stats_map, COL_LIST, KWARGS_MAP) 
    else:
        df = gen_summary_stats(df, col_stats_pairs, root_stats_map, COL_LIST, KWARGS_MAP)

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
