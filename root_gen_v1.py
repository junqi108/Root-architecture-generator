#!/usr/bin/python

##########################################################################################################
### Root Generation Version 1

# This is the initial implementation of the synthetic root generation script.
# Several additional features and alterations were suggested following consultation.
# The suggestions were drastic enough that it was therefore decided to version the script.
##########################################################################################################

##########################################################################################################
### Imports
##########################################################################################################

import argparse
import math
import numpy as np
import os
import pandas as pd
import random
from numpy.random import uniform

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Simulate a root system.'
)

def _add_argument(parser: argparse.ArgumentParser, name: str, default: str, arg_help: str, type = str) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )

# Ouput
_add_argument(parser, "--dir", "data/root_sim", "Output directory")
_add_argument(parser, "--out", "root_sim.csv", "Output CSV file name")
_add_argument(parser, "--round", 4, "Number of decimal places to round values when writing to file", int)
_add_argument(parser, "--stats", 1, "Calculate statistics for the root system", int)
_add_argument(parser, "--bins", 10, "Number of bins when binning root data", int)

# Roots
_add_argument(parser, "--order", 2, "The maximum root order", int)
## Primary
_add_argument(parser, "--rnum_out", 10, "The number of outer primary roots to be generated", int)
_add_argument(parser, "--rnum_in", 5, "The number of inner primary roots to be generated", int)
### Size
_add_argument(parser, "--min_prlength", 200, "The minimum length of each primary root in centimetres", float)
_add_argument(parser, "--max_prlength", 300, "The maximum length of each primary root in centimetres", float)
### In ArchiSimple model, root diameter is in metres ###
_add_argument(parser, "--min_pdiam", 0.009, "The minimum diameter of primary roots in millimetres", float)
_add_argument(parser, "--max_pdiam", 0.012, "The maximum diameter of primary roots in millimetres", float)
## Secondary
_add_argument(parser, "--srnum_min", 0, "The minimum number of secondary roots to be generated", int)
_add_argument(parser, "--srnum_max", 5, "The maximum number of secondary roots to be generated", int)
### Size
_add_argument(parser, "--min_srlength", 25, "The minimum length of each secondary root in centimetres", float)
_add_argument(parser, "--max_srlength", 100, "The maximum length of each secondary root in centimetres", float)
### In ArchiSimple model, root diameter is in metres ###
_add_argument(parser, "--min_sdiam", 0.004, "The minimum diameter of secondary roots in millimetres", float)
_add_argument(parser, "--max_sdiam", 0.007, "The maximum diameter of secondary roots in millimetres", float)

# Segments
_add_argument(parser, "--snum", 50, "The number of segments per root", int)
_add_argument(parser, "--fix_seg", 1, "Use a fixed segment size for each root", int)
## Size
_add_argument(parser, "--min_pslength", 0.05, "The minimum length of each primary root segment in centimetres", float)
_add_argument(parser, "--max_pslength", 0.1, "The maximum length of each primary root segment in centimetres", float)
_add_argument(parser, "--min_sslength", 0.015, "The minimum length of each secondary root segment in centimetres", float)
_add_argument(parser, "--max_sslength", 0.025, "The maximum length of each secondary root segment in centimetres", float)

# Movement
_add_argument(parser, "--xy_vary", 5, "Random variation in degrees of subsequent segments along x and y axes", int)
_add_argument(parser, "--z_vary", 5, "Random variation in degrees of subsequent segments along z axis", int)

args = parser.parse_args()

##########################################################################################################
### Constants
##########################################################################################################

# Ouput
DIR = args.dir
OUT = f"{DIR}/{args.out}"
ROUND = args.round
CALC_STATS = bool(args.stats)
STATS_OUT = f"{DIR}/root_stats.csv"
BINS = args.bins

# Roots
MAX_ORDER = args.order
## Primary
OUT_ROOTNUM = args.rnum_out
IN_ROOTNUM = args.rnum_in
### Size
MIN_PRLENGTH = args.min_prlength
MAX_PRLENGTH = args.max_prlength
MIN_PDIAMETER = args.min_pdiam
MAX_PDIAMETER = args.max_pdiam
## Secondary
MIN_SRNUM = args.srnum_min
MAX_SRNUM = args.srnum_max
### Size
MIN_SRLENGTH = args.min_srlength
MAX_SRLENGTH = args.max_srlength
MIN_SDIAMETER = args.min_sdiam
MAX_SDIAMETER = args.max_sdiam

# Segments
SEGNUM = args.snum
SEGNUM_FRACTION = int(SEGNUM / 3)
SEGNUM_INTERVAL = int(SEGNUM * 0.1)
FIXED_SEG_LENGTH = bool(args.fix_seg)
## Size
MIN_PSLENGTH = args.min_pslength
MAX_PSLENGTH = args.max_pslength
MIN_SSLENGTH = args.min_sslength
MAX_SSLENGTH = args.max_sslength

# Movement
XY_VARY = args.xy_vary
Z_VARY = args.z_vary
# Set origin point of root system
# The base node is located at origin (0, 0, 0)
# We include a small deviation from this origin for all child nodes to prevent their dimensions from being exactly 0
origin = 1E-1
START_X = float(origin)
START_Y = float(origin)
START_Z = float(origin)

##########################################################################################################
### Main
##########################################################################################################

def get_cart_coords(theta : int, adj_length: float) -> tuple:
    """Calculate the cartesian coordinates of the new segment"""
    if (0.0 < theta < 90.0):
        x = math.cos(math.radians(theta)) * adj_length
        y = math.sin(math.radians(theta)) * adj_length
    elif theta == 90.0:
        x = 0
        y = adj_length
    elif (90.0 < theta < 180.0):
        theta = 180.0 - theta
        x = -(math.cos(math.radians(theta)) * adj_length)
        y = math.sin(math.radians(theta)) * adj_length
    elif theta == 180.0:
        x = -adj_length
        y = 0
    elif (180.0 < theta < 270.0):
        theta = theta - 180.0
        x = -(math.cos(math.radians(theta)) * adj_length)
        y = -(math.sin(math.radians(theta)) * adj_length)
    elif theta == 270.0:
        x = 0
        y = -adj_length
    elif (270.0 < theta < 360.0):
        theta = 360.0 - theta
        x = math.cos(math.radians(theta)) * adj_length
        y = -(math.sin(math.radians(theta)) * adj_length)
    else:
        x = adj_length
        y = 0
    return x, y

def generate_roots(rows: list, order: int, start_angle: float, root: int, 
    phi_angles: int, min_length : float, max_length : float, start_x : float, 
    start_y: float, start_z: float, diameters, root_bend = {},
    parent = 1) -> None:
    """Generate random root system"""
    theta_angles = [root * start_angle] #Seperate roots by equal angles at start
    root_coords = [{"x" : start_x, "y" : start_y, "z" : start_z}]

    if FIXED_SEG_LENGTH:
        r_length = uniform(min_length, max_length)
        seg_length = r_length / SEGNUM
        
    # Override default angle variance for specified root segments
    root_bend_intervals = {}
    for segment in root_bend.keys():
        for interval_key in range(segment - SEGNUM_INTERVAL, segment + SEGNUM_INTERVAL + 1):
            root_bend_intervals[interval_key] = root_bend[segment]

    # Randomly select start and end root segment diameters
    diameter_diff = 0
    while diameter_diff < abs(diameters[1] - diameters[0]) * 0.15:
        # Sort diameters from largest to smallest
        root_diameters = -np.sort(-uniform(diameters[0], diameters[1], 2))
        diameter_diff = abs(root_diameters[1] - root_diameters[0])

    # Apply linear interpolation between start and end segment diameters and add random noise
    diameter_diff *= 0.025
    # Sort random noise from largest to smallest as the root should become progressively become thinner
    diameter_noise = -np.sort(-uniform(-diameter_diff, diameter_diff, SEGNUM))
    diameters = np.interp(range(1, SEGNUM + 1), [1, SEGNUM], root_diameters) + diameter_noise
    
    for segment in range(0, SEGNUM):
        if segment not in root_bend_intervals:
            #Vary the angle of subsequent segments by +-Z degrees in horizontal and vertical directions
            phi = random.randint(-Z_VARY, Z_VARY) 
            # if phi > 0:
            #     phi = -phi
        else:
            root_bend_interval = root_bend_intervals[segment]
            phi = random.randint(root_bend_interval[0], root_bend_interval[1]) 
        
        phi += phi_angles[segment]  
        phi_angles.append(phi)

        if FIXED_SEG_LENGTH is False:
            seg_length = uniform(min_length, max_length)
        z = math.sin(math.radians(phi)) * seg_length
        adj_length = math.cos(math.radians(phi)) * seg_length
        theta = random.randint(-XY_VARY, XY_VARY) + theta_angles[segment]
        if theta > 360:
            theta = theta - 360
        theta_angles.append(theta)
            
        x, y = get_cart_coords(theta, adj_length)
        current_coord = root_coords[segment]
        # Add a small amount of random noise to coordinates to prevent identical values
        noise = uniform(0.0001, 0.00011)
        new_coord = {
            "x" : current_coord["x"] + x + noise, 
            "y" : current_coord["y"] + y + noise,
            "z" : current_coord["z"] + z + noise
        }

        #if new_coord["z"] > 0:
        #    new_coord["z"] = new_coord["z"] - new_coord["z"] -  uniform(0.1, 1)
           # if phi_angles[segment + 1]  > 0:
           #     phi_angles[segment + 1] = -phi_angles[segment + 1]

        root_coords.append(new_coord)

        if segment  == 0:
            parent_root = parent
        else:
            parent_root = root * SEGNUM + segment + 1

        row = { 
            "organ_id" : root + 1,
            "order": order,
            "segment_rank" : segment + 1,
            "parent": parent_root ,
            "coordinates" : ' '.join([str(round(value, ROUND)) for value in current_coord.values()]),
            "diameter" : round(diameters[segment], ROUND), 
            "length" : round(seg_length, ROUND),
            "x": current_coord["x"],
            "y": current_coord["y"],
            "z": current_coord["z"],
        }
        rows.append(row)

def get_parent_coords(rows: list, root: int, segnum : int, n_roots: int, floor = 0.9, ceiling = 0.05) -> list:
    """Sample segments from the parent root and return their coordinates."""
    parent_root_idx = (root + 1) * segnum

    #parent_segments = rows[int(parent_root_idx - segnum * floor) : int(parent_root_idx - segnum * ceiling)]
    #segment_samples = random.sample(parent_segments, n_roots)
    segment_indices = random.sample(range(int(parent_root_idx - segnum * floor), int(parent_root_idx - segnum * ceiling)), n_roots) 
    parent_segments = np.asarray(rows) 
    segment_samples = parent_segments[segment_indices]
    coords = []
    for segment in segment_samples:
        start_coordinates = segment["coordinates"].split(" ")
        start_coordinates = [float(coord) for coord in start_coordinates]
        coords.append(start_coordinates)
    return segment_indices, coords

def get_df_coords(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Get 3D coordinates from dataframe"""
    return (df[col]
    .str
    .split(' ', expand = True)
    .astype(float)
    .rename(columns = {0 : "x", 1 : "y", 2 : "z"}))

def root_cum_sum(coord: np.array, bins = 10) -> np.array:
    """Compute the cumulative distribution function for root coordinates."""
    count, bins_count = np.histogram(coord, bins = bins)
    bins_count = np.insert(bins_count, 0, 0)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)
    return cdf, bins_count[1:]

def calc_root_stats(df : pd.DataFrame, out : str) -> None:
    """Calculate root statistics."""
    coords = get_df_coords(df, "coordinates")
    depth = abs(coords.z)
    horizontal = abs(coords.melt(value_vars=["x", "y"]).value)
    depth_density, depth_bin = root_cum_sum(depth, BINS)
    horizontal_density, horizontal_bin = root_cum_sum(horizontal, BINS)
    stats_df = pd.DataFrame({
        "depth_cum": depth_density,
        "depth_bin": depth_bin,
        "horizontal_cum": horizontal_density,
        "horizontal_bin": horizontal_bin
    })
    stats_df.to_csv(out, index = False)
    print(f"Stats file written to {out}")

def main() -> None:
    """Generate random root structure"""

    print("Generating random root structure...\n")
    rows = [{ 
        "organ_id" : 0,
        "order": 0,
        "segment_rank" : 0,
        "parent": -1,
        "coordinates" : '0 0 0',
        "diameter" : MAX_PDIAMETER, 
        "length" : MAX_PSLENGTH,
        "x": 0,
        "y": 0,
        "z": 0
    }]

    # Primary roots
    ## Outer
    start_angle = 360 / OUT_ROOTNUM
    end = OUT_ROOTNUM
    root_bend = {
        SEGNUM_FRACTION : [-Z_VARY, 0],
        2 * SEGNUM_FRACTION : [0, Z_VARY]
    }
    
    if FIXED_SEG_LENGTH:
        min_length, max_length = MIN_PRLENGTH, MAX_PRLENGTH
    else:
        min_length, max_length = MIN_PSLENGTH, MAX_PSLENGTH

    for root in range(0, end):
        phi_angles = [-(random.randint(10, 15))] #Generates starting downwards angle of the root
        generate_roots(rows, 1, start_angle, root, phi_angles, min_length, max_length,
        START_X, START_Y, START_Z, [MIN_PDIAMETER, MAX_PDIAMETER], root_bend = {})

    ## Inner
    start_angle = 360 / IN_ROOTNUM
    start = end
    end += IN_ROOTNUM
    for root in range(start, end):
        phi_angles = [-random.randint(20, 90)] 
        root_bend = None
        if phi_angles[0] < 60:
            root_bend = { SEGNUM_FRACTION : [0, Z_VARY] }
        generate_roots(rows, 1, start_angle, root, phi_angles, min_length, max_length,
        START_X, START_Y, START_Z, [MIN_PDIAMETER, MAX_PDIAMETER], root_bend = {})
    
    # Secondary roots
    roots = list(range(0, end))
    if FIXED_SEG_LENGTH:
        min_length, max_length = MIN_SRLENGTH, MAX_SRLENGTH
    else:
        min_length, max_length = MIN_SSLENGTH, MAX_SSLENGTH

    for order in range(2, MAX_ORDER + 1):
        secondary_roots = []
        for root in roots:
            n_roots = random.randint(MIN_SRNUM, MAX_SRNUM)
            if n_roots == 0:
                continue

            parent_indices, parent_coords = get_parent_coords(rows, root, SEGNUM, n_roots)
            start_angle = 360 / n_roots

            start = end
            end += n_roots
            for i, secondary_root in enumerate(range(start, end)):
                secondary_roots.append(secondary_root)
                parent_coord = parent_coords[i]
                phi_angles = [-(random.randint(0, 360))]  
                generate_roots(rows, order, start_angle, secondary_root, phi_angles, min_length, max_length,
                parent_coord[0], parent_coord[1], parent_coord[2], [MIN_SDIAMETER, MAX_SDIAMETER], parent = parent_indices[i])

        roots = secondary_roots
 
    os.makedirs(DIR, exist_ok = True)
    df = pd.DataFrame(rows)
    df.index += 1 
    df.to_csv(OUT, index = True, index_label = "id")
    print(f"File written to {OUT}")

    if CALC_STATS:
        calc_root_stats(df, STATS_OUT)
        
if __name__ == "__main__":
    main()