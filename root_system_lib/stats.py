"""
Root System Library

A library of methods for calculating statistics pertaining to synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple

# Internal
from root_system_lib.constants import ROOT_TYPES
from root_system_lib.soil import make_soil_grid

##########################################################################################################
### Library
##########################################################################################################

def get_df_coords(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Get 3D coordinates from dataframe

    Parameters
    --------------
    df: DataFrame
        The dataframe of root data.
    col: str
        The column name.

    Returns
    ---------
    coordinates: DataFrame
        The root coordinates.
    """
    return (df[col]
    .str
    .split(' ', expand = True)
    .astype(float)
    .rename(columns = {0 : "x", 1 : "y", 2 : "z"}))

def root_cum_sum(coord: np.array, bins = 10) -> np.array:
    """
    Compute the cumulative density function for root coordinates.
    
    Parameters
    --------------
    coord: array (n, 3)
        The 3D root coordinates.
    bins: int
        The number of bins when discretising data.

    Returns
    ---------
    cdf, bins: tupple
        The cumulative distribution function and bins.
    """
    count, bins_count = np.histogram(coord, bins = bins)
    bins_count = np.insert(bins_count, 0, 0)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)
    return cdf, bins_count[1:]

def calc_root_stats(df : pd.DataFrame, out : str, bins: int = 10) -> None:
    """
    Calculate root statistics.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    out: str
        The output file path.
    bins: int
        The number of bins when discretising data.
    """
    coords = get_df_coords(df, "coordinates")
    depth = abs(coords.z)
    horizontal = abs(coords.melt(value_vars=["x", "y"]).value)
    depth_density, depth_bin = root_cum_sum(depth, bins)
    horizontal_density, horizontal_bin = root_cum_sum(horizontal, bins)
    stats_df = pd.DataFrame({
        "depth_cum": depth_density,
        "depth_bin": depth_bin,
        "horizontal_cum": horizontal_density,
        "horizontal_bin": horizontal_bin
    })
    stats_df.to_csv(out, index = False)
    print(f"Stats file written to {out}")

def calc_root_depth(df: pd.DataFrame, depth_cum_col: str, bins: int = 10, values_only: bool = False) -> Tuple[float]:
    """
    Calculate the cumulative root depth.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    depth_cum_col: str
        The dataframe column to calculate values.
    bins: int
        The number of bins when discretising data.
    values_only: bool
        Only include the cumulative root depth, and not the cumulative proportions assigned to each bin.

    Returns
    ---------
    cdf, bins: tuple
        The cumulative distribution function and bins.
    """
    depth = abs(df[depth_cum_col])
    cum_sum_values = np.array(root_cum_sum(depth, bins))
    if values_only:
        cum_sum_values = cum_sum_values[1]
    return cum_sum_values

def calc_root_horiz_distance(df: pd.DataFrame, horizontal_cum_cols: list, bins: int = 10, values_only: bool = False) -> Tuple[float]:
    """
    Calculate the cumulative horizontal root distance.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    horizontal_cum_cols: str
        The dataframe column to calculate values.
    bins: int
        The number of bins when discretising data.
    values_only: bool
        Only include the cumulative root distance, and not the cumulative proportions assigned to each bin.

    Returns
    ---------
    cdf, bins: tuple
        The cumulative distribution function and bins.
    """
    horizontal = abs(df.melt(value_vars=horizontal_cum_cols).value)
    cum_sum_values = np.array(root_cum_sum(horizontal, bins))
    if values_only:
        cum_sum_values = cum_sum_values[1]
    return cum_sum_values

def get_points_inside_cube(cube: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """
    Determine the location of each point with respect to each soil block.
    
    Parameters
    --------------
    cube: DataFrame
        The soil cube dataframe.
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    df_res: DataFrame
        The dataframe query results for whether each point is located within the soil block.
    """
    df_query = ""
    cube_dict = {}
    
    def _build_query(df_query: str, col: str, cube_dict, add_amp: bool = True) -> str:
        col_min, col_max = np.min(cube[col]), np.max(cube[col])
        df_query += f"{col} > {col_min} & {col} < {col_max}"
        if add_amp:
            df_query += " & "

        cube_dict[f"{col}_min"], cube_dict[f"{col}_max"] = col_min, col_max
        return df_query, cube_dict
    
    df_query, cube_dict = _build_query(df_query, "x", cube_dict)
    df_query, cube_dict = _build_query(df_query, "y", cube_dict)
    df_query, cube_dict = _build_query(df_query, "z", cube_dict, False)
    df_res = df.query(df_query)

    pd.options.mode.chained_assignment = None
    for k, v in cube_dict.items():
        df_res[k] = v
    return df_res

def __get_roots_per_vol(df: pd.DataFrame, soil_block_size: float) -> pd.DataFrame:
    soil_grid = make_soil_grid(df, soil_block_size)
    roots_per_vol = soil_grid.groupby("block_id").apply(get_points_inside_cube, df)
    return roots_per_vol

def calc_root_length_density_depth(df: pd.DataFrame, soil_block_size: float, as_scalar: bool = True) -> pd.DataFrame:
    """
    Calculate the root length density with respect to soil depth.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    soil_block_size: float
        The size of each soil block.
    as_scalar: bool
        Convert the RLD to scalar value.

    Returns
    ---------
    rld_df: DataFrame
        The dataframe of root length density calculations.
    """
    roots_per_vol = __get_roots_per_vol(df, soil_block_size)
    rld_df = roots_per_vol.groupby(["z_min", "z_max"]).apply(lambda row: row.length.sum()).sort_values(ascending = False)
    if as_scalar:
        rld_df = rld_df.mean()
    return rld_df

def calc_average_root_length_density_depth(df: pd.DataFrame, soil_block_size: float, as_scalar: bool = True) -> pd.DataFrame:
    """
    Calculate the average root length density with respect to soil depth.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    soil_block_size: float
        The size of each soil block.
    as_scalar: bool
        Convert the RLD to scalar value.

    Returns
    ---------
    rld_df: DataFrame
        The dataframe of average root length density calculations.
    """
    roots_per_vol = __get_roots_per_vol(df, soil_block_size)
    rld_df = roots_per_vol.groupby(["z_min", "z_max"]).apply(lambda row: row.length.mean()).sort_values(ascending = False)
    if as_scalar:
        rld_df = rld_df.mean()
    return rld_df

def calc_root_weight_density_depth(df: pd.DataFrame, soil_block_size: float, rtd: float, as_scalar: bool = True) -> pd.DataFrame:
    """
    Calculate the root weight density with respect to soil depth.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    soil_block_size: float
        The size of each soil block.
    rtd: float
        The root tissue density.
    as_scalar: bool
        Convert the RLD to scalar value.

    Returns
    ---------
    rwd_df: DataFrame
        The dataframe of root length density calculations.
    """
    roots_per_vol = __get_roots_per_vol(df, soil_block_size)
    rwd_df = roots_per_vol.groupby(["z_min", "z_max"]).apply(calc_root_total_weight, rtd).sort_values(ascending = False)
    if as_scalar:
        rwd_df = rwd_df.mean()
    return rwd_df

def calc_average_root_weight_density_depth(df: pd.DataFrame, soil_block_size: float, rtd: float, as_scalar: bool = True) -> pd.DataFrame:
    """
    Calculate the average root weight density with respect to soil depth.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    soil_block_size: float
        The size of each soil block.
    rtd: float
        The root tissue density.
    as_scalar: bool
        Convert the RLD to scalar value.

    Returns
    ---------
    rwd_df: DataFrame
        The dataframe of average root length density calculations.
    """
    roots_per_vol = __get_roots_per_vol(df, soil_block_size)
    rwd_df = roots_per_vol.groupby(["z_min", "z_max"]).apply(calc_root_average_weight, rtd).sort_values(ascending = False)
    if as_scalar:
        rwd_df = rwd_df.mean()
    return rwd_df

def __calc_specific_root_length(df: pd.DataFrame, rtd: float) -> np.ndarray:
    total_volume = calc_root_total_volume(df)
    mass = total_volume * rtd
    srl = df.length / mass
    return srl

def calc_total_specific_root_length(df: pd.DataFrame, rtd: float) -> float:
    """
    Calculate the total specific root length.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    rtd: float
        The root tissue density.

    Returns
    ---------
    srl: float
        The total specific root length.
    """
    srl = __calc_specific_root_length(df, rtd)
    return np.sum(srl)

def calc_average_specific_root_length(df: pd.DataFrame, rtd: float) -> float:
    """
    Calculate the average specific root length.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    rtd: float
        The root tissue density.

    Returns
    ---------
    srl: float
        The average specific root length.
    """
    total_srl = calc_total_specific_root_length(df, rtd)
    return __calc_avg(df, total_srl)

def calc_root_total_volume(df: pd.DataFrame) -> float:
    """
    Calculate the total root volume, where each root segment is treated as a cylinder.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    rv: float
        The total root volume.
    """
    radius = df.diameter / 2
    height = df.length
    volume = np.pi * radius**2 * height
    return np.sum(volume)

def calc_root_average_volume(df: pd.DataFrame) -> float:
    """
    Calculate the average root volume, where each root segment is treated as a cylinder.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    rv: float
        The average root volume.
    """
    total_volume = calc_root_total_volume(df)
    return __calc_avg(df, total_volume)

def calc_root_total_weight(df: pd.DataFrame, rtd: float) -> float:
    """
    Calculate the total root weight, where each root segment is treated as a cylinder.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    rtd: float
        The root tissue density.

    Returns
    ---------
    weight: float
        The total root weight.
    """
    total_volume = calc_root_total_volume(df)
    total_weight = total_volume * rtd
    return total_weight

def calc_root_average_weight(df: pd.DataFrame, rtd: float) -> float:
    """
    Calculate the average root weight, where each root segment is treated as a cylinder.
    
    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    rtd: float
        The root tissue density.

    Returns
    ---------
    weight: float
        The average root weight.
    """
    total_weight = calc_root_total_weight(df, rtd)
    return __calc_avg(df, total_weight)

def calc_root_total_length(df: pd.DataFrame) -> float:
    """
    Calculate the total root length.

    Parameters
    --------------
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    length: float
        The total root length.
    """
    return __calc_total(df, "length")

def calc_root_total_diameter(df: pd.DataFrame) -> float:
    """
    Calculate the total root diameter.

    Parameters
    --------------
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    diameter: float
        The total root diameter.
    """
    return __calc_total(df, "diameter")

def calc_root_average_length(df: pd.DataFrame) -> float:
    """
    Calculate the average root length.

    Parameters
    --------------
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    length: float
        The average root length.
    """
    return __calc_avg_from_total(df, "length")

def calc_root_average_diameter(df: pd.DataFrame) -> float:
    """
    Calculate the average root diameter.

    Parameters
    --------------
    df: DataFrame
        The root dataframe.

    Returns
    ---------
    diameter: float
        The average root diameter.
    """
    return __calc_avg_from_total(df, "diameter")

def __calc_avg(df, total: float) -> float:
    n = len(df.organ_id.unique()) - 1
    avg = total / n
    return avg

def __calc_total(df: pd.DataFrame, col: str) -> float:
    total = np.sum(df[col])
    return total

def __calc_avg_from_total(df: pd.DataFrame, col: str) -> float:
    total = __calc_total(df, col)
    return __calc_avg(df, total)

def get_root_stats_map() -> tuple:
    """Get a map of available root statistics, and associated functions and arguments."""
    root_stats_map = {
        "depth_cum": {"func": calc_root_depth, "kwarg_keys": ["bins", "depth_cum_col", "values_only"]},
        "horizontal_cum": {"func": calc_root_horiz_distance, "kwarg_keys": ["bins", "horizontal_cum_cols", "values_only"]},
        "rld_depth" : {"func": calc_root_length_density_depth, "kwarg_keys": ["soil_block_size", "as_scalar"]},
        "average_rld_depth" : {"func": calc_average_root_length_density_depth, "kwarg_keys": ["soil_block_size", "as_scalar"]},
        "rwd_depth" : {"func": calc_root_weight_density_depth, "kwarg_keys": ["soil_block_size", "rtd", "as_scalar"]},
        "average_rwd_depth" : {"func": calc_average_root_weight_density_depth, "kwarg_keys": ["soil_block_size", "rtd", "as_scalar"]},
        "total_srl" : {"func": calc_total_specific_root_length, "kwarg_keys": ["rtd"]},
        "average_srl" : {"func": calc_average_specific_root_length, "kwarg_keys": ["rtd"]},
        "total_volume" : {"func": calc_root_total_volume, "kwarg_keys": []},
        "average_volume" : {"func": calc_root_average_volume, "kwarg_keys": []},
        "total_weight" : {"func": calc_root_total_weight, "kwarg_keys": ["rtd"]},
        "average_weight" : {"func": calc_root_average_weight, "kwarg_keys": ["rtd"]},
        "total_length" : {"func": calc_root_total_length, "kwarg_keys": []},
        "total_diameter" : {"func": calc_root_total_diameter, "kwarg_keys": []},
        "average_length" : {"func": calc_root_average_length, "kwarg_keys": []},
        "average_diameter" : {"func": calc_root_average_diameter, "kwarg_keys": []},
    }

    available_stats = list(root_stats_map.keys())
    return root_stats_map, available_stats

def exec_root_stats_map(df, root_stats_map: Dict[str, dict], root_stats: List[str], kwargs_map: dict, root_type: str = None) -> dict:
    """
    Execute root statistic functions, pass arguments from the kwargs map, and return the mapped results. 

    Parameters
    --------------
    df: DataFrame
        The root dataframe.
    root_stats_map: dict
        The dictionary of available root stats.
    root_stats: list
        The list of requested root stats.
    kwargs_map: dict
        A map of arguments passed to each statistic.
    root_type: str
        The root type classification: Fine or structural roots.

    Returns
    ---------
    map: dict
        The map of calculated root statistics.
    """
    func_results = {}

    if root_type:
        root_type_id = ROOT_TYPES.index(root_type) + 1
        df = df.query(f"root_type == {root_type_id}")

    for root_stat in root_stats:
        # Get root stats definition from root_stats_map - get_root_stats_map()
        stats_definition = root_stats_map.get(root_stat)
        if stats_definition is None:
            raise Exception(f"Root statistic does not exist: {root_stat}")
        # Build kwargs for root stats function
        kwargs = {k: kwargs_map.get(k) for k in stats_definition["kwarg_keys"]}
        # Call function and add results
        func_results[root_stat] = stats_definition["func"](df, **kwargs)
    return func_results
        
def read_stats_data(calc_statistics: bool, obs_file: str, stats_file: str, root_stats_list: list, kwargs_map: dict, 
    col_stats_map: list = None, root_type: str = None):
    """
    Read either an observations file or stats file. Converts the resulting dataframe into a dictionary.
    
    Parameters
    --------------
    calc_statistics: bool
        Whether or not to calculate root statistics from the observation file.
    obs_file: str
        The file path of the observation CSV file.
    stats_file: str
        The file path of the root statistics CSV file.
    root_stats_list: list
        The list of requested root stats.
    kwargs_map: dict
        A map of arguments passed to each statistic.
    col_stats_map: list
        A list of root statistics and their mapped column names.
    root_type: str
        The root type classification: Fine or structural roots.

    Returns
    ---------
    map: dict
        The map of calculated root statistics.
    """
    root_stats_map, _ = get_root_stats_map()

    if calc_statistics:
        obs_df = pd.read_csv(obs_file)
        obs_statistics = exec_root_stats_map(obs_df, root_stats_map, root_stats_list, kwargs_map, root_type)
    else:
        stats_df = pd.read_csv(stats_file)
        obs_statistics = {}

        if col_stats_map is None:
            for k in root_stats_list:
                stat_values = stats_df[k].dropna().values
                obs_statistics[k] = stat_values
        else:
            col_stats_list = [ col_stat.split("=") for col_stat in col_stats_map.split(",") ]
            col_stats_dict = { k: v for k, v in col_stats_list }
            for k in root_stats_list:
                column_name = col_stats_dict[k]
                stat_values = stats_df[column_name].dropna().values           
                obs_statistics[k] = stat_values

    return obs_statistics, root_stats_map