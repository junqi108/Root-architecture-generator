"""
Soil Library

A library of methods for synthetic soil generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import pandas as pd
import numpy as np

##########################################################################################################
### Library
##########################################################################################################

def make_soil_block(block_dim: float) -> np.ndarray:
    """
    Create a single soil block.

    Parameters
    --------------
    block_dim: float
        The dimensionality of a single soil block

    Returns
    ---------
    block : array  
        The soil block.
    """
    dim = [0, block_dim]
    soil_block = np.array(np.meshgrid(dim, dim, dim)).T.reshape(-1, 3)
    return soil_block

def calc_n_blocks(block_dim: float, bound: list) -> tuple:
    """
    Calculate the number of soil blocks needed to encapsulate the root system.

    Parameters
    --------------
    block_dim: float
        The dimensionality of a single soil block
    bound: list
        The lower and upper boinds of the root system.

    Returns
    ---------
    n_min, n_max : tuple (2,)  
        The minimum and maximum number of soil blocks.
    """
    min_bound, max_bound = bound
    n_min = abs(np.ceil(min_bound / block_dim)) + 1
    n_max = abs(np.ceil(max_bound / block_dim)) 
    return n_min, n_max

def build_soil_grid(base_block: np.ndarray, block_dim: float, x_bounds: tuple, y_bounds: tuple, z_bounds: tuple) -> np.ndarray:
    """
    Construct a grid of soil blocks.

    Parameters
    --------------
    base_block: array (n, d)
        The single base soil block.
    block_dim: float
        The dimensionality of a single soil block
    x_bounds: tuple
        The lower and upper bound of the x dimension.
    y_bounds: tuple
        The lower and upper bound of the y dimension.
    z_bounds: tuple
        The lower and upper bound of the z dimension.

    Returns
    ---------
    grid: tuple (n,d)  
        The soil grid.
    """
    x_n_min, x_n_max = x_bounds
    y_n_min, y_n_max = y_bounds
    z_n_min, _ = z_bounds

    # Translate and reflect soil block rows
    def _affline_transform_row(replicates, dim, constant = 1):
        for i in range(len(replicates)):
            replicates[i][:, dim] += i * block_dim
            replicates[i][:, dim] *= constant

    # Construct a single row of soil blocks along x-axis
    x_pos_replicates = np.repeat(base_block[None, ...], x_n_max, axis = 0)
    x_neg_replicates = np.repeat(base_block[None, ...], x_n_min, axis = 0)
    _affline_transform_row(x_pos_replicates, 0)
    _affline_transform_row(x_neg_replicates, 0, -1)

    # Merge the two x-axis replicates into a single row
    x_row = np.vstack((x_pos_replicates, x_neg_replicates))
    x_row = x_row.reshape(-1, x_row.shape[-1]) # 3D => 2D

    # Replicate x row downwards towards z-axis
    z_replicates = np.repeat(x_row[None,...], z_n_min, axis = 0)
    _affline_transform_row(z_replicates, 2, -1)
    z_row = z_replicates.reshape(-1, z_replicates.shape[-1])

    # Replicate z row across y-axis
    y_pos_replicates = np.repeat(z_row[None, ...], y_n_max, axis = 0)
    y_neg_replicates = np.repeat(z_row[None, ...], y_n_min, axis = 0)
    _affline_transform_row(y_pos_replicates, 1)
    _affline_transform_row(y_neg_replicates, 1, -1)
    y_row = np.vstack((y_pos_replicates, y_neg_replicates))
    soil_grid = y_row.reshape(-1, y_row.shape[-1])

    return soil_grid

def make_soil_grid(df: pd.DataFrame, block_dim: float) -> pd.DataFrame:
    """
    Make a grid of soil blocks. Calculate the bounds for the x,y, and z dimensions.

    Parameters
    --------------
    df: DataFrame
        The dataframe of root data.
    block_dim: float
        The dimensionality of a single soil block

    Returns
    ---------
    grid: DataFrame
        The soil grid dataframe.
    """
    def get_min_max(df, col):
        return np.min(df[col]), np.max(df[col])

    x_bounds = get_min_max(df, "x")
    y_bounds = get_min_max(df, "y")
    z_bounds = get_min_max(df, "z")

    # Construct grid
    soil_block = make_soil_block(block_dim)
    x_n_blocks = calc_n_blocks(block_dim, x_bounds)
    y_n_blocks = calc_n_blocks(block_dim, y_bounds)
    z_n_blocks = calc_n_blocks(block_dim, z_bounds)
    soil_grid = build_soil_grid(soil_block, block_dim, x_n_blocks, y_n_blocks, z_n_blocks)
    grid_df = pd.DataFrame(soil_grid, columns = ["x", "y", "z"])

    n_corners = 8
    n_blocks = soil_grid.shape[0] / n_corners
    soil_block_ids = np.repeat(np.arange(n_blocks), n_corners)
    grid_df["block_id"] = soil_block_ids
    
    return grid_df
