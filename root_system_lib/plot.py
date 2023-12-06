"""
Root System Library

A library for plotting data from synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from matplotlib import cm

##########################################################################################################
### Library
##########################################################################################################

def visualise_roots(df: pd.DataFrame, thickness: int = 4, include_properties = False, soil_grid = None) -> None:
    """
    Provide a 3D visualisation of the root system.
    
    Parameters
    --------------
    df: DataFrame
        The dataframe.
    thickness: int
        The line thickness.
    include_properties: bool
        Include the dataframe properties.
    soil_grid: bool
        Render a soil grid.
    """
    fig = go.Figure()

    if include_properties:
        df.groupby("organ_id").apply(
            lambda root: fig.add_trace(
                go.Scatter3d(
                    x = root.x, 
                    y = root.y, 
                    z = root.z,
                    marker = dict(size = thickness),
                    line = dict(color='green', width = thickness), 
                    customdata = np.stack((root.organ_id, root.order, root.segment_rank, root.diameter, root.length), axis=-1),
                    hovertemplate =
                    'x: %{x}<br>'+
                    'y: %{y}<br>'+
                    'z: %{z}<br>'+
                    'root_id: %{customdata[0]}<br>'+
                    'order: %{customdata[1]}<br>'+
                    'segment_rank: %{customdata[2]}<br>'+
                    'diameter: %{customdata[3]}<br>' + 
                    'length: %{customdata[4]}<br>'
                )
            )
        )
    else:
        df.groupby("organ_id").apply(
            lambda root: fig.add_trace(
                go.Scatter3d(
                    x = root.x, 
                    y = root.y, 
                    z = root.z,
                    marker = dict(size = thickness),
                    line = dict(color='green', width = thickness)
                )
            )
        )

    if soil_grid is not None:
        fig.add_trace(
            go.Scatter3d(
                x = soil_grid.x, 
                y = soil_grid.y, 
                z = soil_grid.z,
                mode='markers',
                marker = dict(color = "blue")
            )
        )

    fig.layout.update(showlegend = False) 
    fig.show()

def root_dist_depth(stats_df: pd.DataFrame) -> None:
    """
    Display the cumulative root distribution by soil depth. 

    Parameters
    --------------
    stats_df: DataFrame
        The dataframe of root stats.
    """
    plt.plot(stats_df.depth_cum, stats_df.depth_bin, '-o', color="blue")
    plt.gca().invert_yaxis()
    plt.title("Cumulative root distribution by soil depth")
    plt.xlabel("Cumulative root fraction") 
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.ylabel("Soil depth (cm)") 
    plt.show()

def root_dist_horizontal(stats_df: pd.DataFrame) -> None:
    """
    Display the cumulative root distribution by horizontal distance from the plant base.

    Parameters
    --------------
    stats_df: DataFrame
        The dataframe of root stats.
    """
    plt.plot(stats_df.horizontal_cum, stats_df.horizontal_bin, '-o', color="blue")
    plt.title("Cumulative root distribution by horizontal distance")
    plt.xlabel("Cumulative root fraction") 
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.ylabel("Horizontal root distance (cm)") 
    plt.show()

def root_dist_depth_horizontal(coords: pd.DataFrame) -> None:
    """
    Plot a histogram of the root distribution with depth against horizontal distance.

    Parameters
    --------------
    coords: DataFrame
        The root coordinates.
    """
    depth = abs(coords.z)
    depth_stacked = np.hstack([depth, depth])
    horizontal = abs(coords.melt(value_vars=["x", "y"]).value)

    fig = plt.figure()          
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(depth_stacked, horizontal, bins=(10,10))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like(xpos)
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    cmap = cm.get_cmap('jet')  
    max_height = np.max(dz)    
    min_height = np.min(dz)
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title("Root depth against horizontal root length")
    ax.set_xlabel("Root depth (cm)")
    ax.set_ylabel("Horizontal root length (cm)")
    ax.set_zlabel("Frequency")
    plt.show()