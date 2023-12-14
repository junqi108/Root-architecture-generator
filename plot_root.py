#!/usr/bin/python3

##########################################################################################################
### Imports
##########################################################################################################

# External
import argparse
import pandas as pd

# Internal
from root_system_lib.config import add_argument, Config
from root_system_lib.plot import root_dist_depth, root_dist_horizontal, root_dist_depth_horizontal, visualise_roots
from root_system_lib.soil import make_soil_grid
from root_system_lib.stats import get_df_coords

##########################################################################################################
### Parameters
##########################################################################################################

def get_parser():
    parser = argparse.ArgumentParser(
        description = 'Visualise the root system.'
    )

    add_argument(parser, "--dir", "data/root_sim", "Input directory", str)
    add_argument(parser, "--plant", None, "The plant to visualise")

    add_argument(parser, "--input", "root_sim.csv", "Input CSV file name", str)
    add_argument(parser, "--stats", 0, "Plot statistics for the root system", choices = [0, 1])
    add_argument(parser, "--soil_grid", 0, "Construct and visualise a soil grid", choices = [0, 1])
    add_argument(parser, "--sblock_size", 1, "The (voxel) size of each soil block in cubic-cm", float)
    add_argument(parser, "--detailed_vis", 0, "Include properties in the root visualisation", choices = [0, 1])
    add_argument(parser, "--thick", 4, "Maximum line thickness for plotted roots")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

##########################################################################################################
### Constants
##########################################################################################################

    CONFIG = Config().from_parser(parser)

    DIR = CONFIG.get("dir")
    PLANT = CONFIG.get("plant")
    INPUT = f"{DIR}/{CONFIG.get('input')}"

    STATS_IN = f"{DIR}/root_stats.csv"
    PLOT_STATS = CONFIG.get_as("stats", bool)

    SOIL_GRID = CONFIG.get_as("soil_grid", bool)
    SOIL_BLOCK_SIZE = CONFIG.get("sblock_size")

    INCLUDE_PROP = CONFIG.get_as("detailed_vis", bool)
    THICKNESS = CONFIG.get("thick")

##########################################################################################################
### Main
##########################################################################################################

def plot_root_stats(coords : pd.DataFrame, stats_name : str) -> None:
    """Visualise root statistics"""
    stats_df = pd.read_csv(stats_name)
    root_dist_depth(stats_df)
    root_dist_horizontal(stats_df)
    root_dist_depth_horizontal(coords)
    
def main() -> None:
    """Visualise root structure"""
    df = pd.read_csv(INPUT)
    if PLANT is not None:
        df = df[df["plant_id"] == PLANT]
    df.drop(["x", "y", "z"], axis = 1, inplace = True)
    start_coords = get_df_coords(df, "coordinates")
    df = pd.concat([df, start_coords], axis = 1)

    soil_grid = None
    if SOIL_GRID:
        soil_grid = make_soil_grid(df, SOIL_BLOCK_SIZE)

    visualise_roots(df, thickness = THICKNESS, include_properties = INCLUDE_PROP, soil_grid = soil_grid)
    if PLOT_STATS:
        end_coords = get_df_coords(df, "coordinates")
        plot_root_stats(end_coords, STATS_IN)
    
if __name__ == "__main__":
    main()