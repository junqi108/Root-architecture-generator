"""
Root System Library

A library of methods relating to spatial properties of synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import math
import matplotlib.pyplot as plt
import numpy as np

##########################################################################################################
### Library
##########################################################################################################

def get_x_rotation_matrix(theta: float) -> np.ndarray:   
    """
    Construct a rotation matrix for the x axis.

    Parameters
    --------------
    theta : float
        The rotation angle in degrees.

    Returns
    ---------
    rotate : (4, 4) float
        The x rotation matrix.
    """

    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))

    x_rotate = np.eye(4)
    x_rotate[1:3, 1:3] = np.array([
        [cos, sin],
        [-sin, cos]
    ])
    return x_rotate

def get_y_rotation_matrix(theta: float) -> np.ndarray:   
    """
    Construct a rotation matrix for the y axis.

    Parameters
    --------------
    theta : float
        The rotation angle in degrees.

    Returns
    ---------
    rotate : (4, 4) float
        The y rotation matrix.
    """
    
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))

    y_rotate = np.eye(4)
    y_rotate[0, 0:3] = [cos, 0, -sin]
    y_rotate[2, 0:3] = [sin, 0, cos]
    return y_rotate

def get_z_rotation_matrix(theta: float) -> np.ndarray:   
    """
    Construct a rotation matrix for the z axis.

    Parameters
    --------------
    theta : float
        The rotation angle in degrees.

    Returns
    ---------
    rotate : (4, 4) float
        The z rotation matrix.
    """
    
    cos = np.cos(np.radians(theta))
    sin = np.sin(np.radians(theta))

    z_rotate = np.eye(4)
    z_rotate[0:2, 0:2] = np.array([
        [cos, sin],
        [-sin, cos]
    ])
    return z_rotate

def transform(roll = 0, pitch = 0, yaw = 0, translation = [0, 0, 0], reflect = [1, 1, 1, 1], scale = [1, 1, 1, 1]):
    """
    Updates the transformation matrix.

    Parameters
    --------------
    roll: int
        Rotation about the x-axis in degrees. 
    pitch: int 
        Rotation about the y-axis in degrees.  
    yaw: int 
        Rotation about the z-axis in degrees. 
    translation: (3,) float
        Translation matrix. 
    reflect: (4,) float
        Reflection matrix. 
    scale: (4,) float
        Scaling matrix. 

    Returns
    ---------
    scale: (4, 4) float
        Transformation matrix.
    """
    # Create transformation matrix
    # Rotations
    x_rotate = get_x_rotation_matrix(roll)
    y_rotate = get_y_rotation_matrix(pitch)
    z_rotate = get_z_rotation_matrix(yaw)
    # Translate
    translate = np.eye(4)
    translate[:-1, 3] = np.array(translation)
    # Affline transformation => 4x4 matrix
    transformation_matrix = translate @ np.diag(scale) @ np.diag(reflect) @ x_rotate @ y_rotate @ z_rotate 
    return transformation_matrix 

def make_homogenous(matrix: np.array) -> np.ndarray:
    """
    Adds an additional 'W' dimension of ones to the matrix.
    Performs a conversion from Cartesian coordinates to homogenous coordinates.

    Parameters
    --------------
    matrix: (n, d) float
        The matrix with Cartesian coordinates.

    Returns
    ---------
    homogenous : (n, d + 1) float
        The homogenous matrix.
    """
    
    ones_matrix = np.ones((len(matrix), 1))
    homogenous_coordinates = np.hstack((matrix, ones_matrix)).T
    return homogenous_coordinates

def degrees_scaler(arr: np.ndarray) -> np.ndarray:
    """
    Scale points to be about 360 degrees.

    Args:
        arr (np.ndarray): 
            The array of points

    Returns:
        np.ndarray: 
            The scaled array of points in degrees.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val) 
    coordinate_arr = scaled_arr * 360
    return coordinate_arr
    
def constrain_points(arr: np.ndarray) -> np.ndarray:
    """
    Constrain points to be within 0-360 degrees.

    Args:
        arr (np.ndarray): 
            The array of points

    Returns:
        np.ndarray: 
            The constrained array of points in degrees.
    """
    arr = np.where(arr < 0, -1 * arr, arr)
    arr = np.where(arr > 360, arr - 360, arr)
    return arr

def sample_half_circle(root_num: tuple, visualise_roots: bool, rng, density: np.ndarray = None):
    """
    Sample yaw and pitch degrees from a half-circle.

    Args:
        root_num (tuple): 
            The number of roots.
        visualise_roots (bool): 
            Whether to visualise the roots.
        rng (RNG): 
            The random number generator.
        density (list): 
            A list of root density proportions per quadrant.

    Returns:
        tuple: 
            The yaw and pitch.
    """
    if visualise_roots:
        root_num = 1000
    elif type(root_num) is tuple:
        root_num = sum(root_num)
    R = 1
    if density:
        yaw, pitch = half_circle_rejection_sampling(density, root_num, R, rng)
    else:
        phi = rng.random(size = root_num) * np.pi 
        r = np.sqrt(rng.random(size = root_num)) * R 
        yaw = r * np.cos(phi) * 180 + 180
        pitch = r * np.sin(phi) * 180 - 45

    if visualise_roots:
        f = plt.figure(figsize=(15, 15))
        a = f.add_subplot(111)
        pitch_vis = -1 * pitch - 45
        a.scatter(yaw, pitch_vis, marker='.')
        a.set_title("Sampling points from a half-circle (5000 samples)")
        a.set_xlabel('Yaw')
        a.set_ylabel('Pitch')
        a.set_aspect("equal")
        a.set_xlim([-20, 380])
        a.set_ylim([-200, 20])
        plt.show()

    return yaw, pitch

def half_circle_rejection_sampling(density: list, root_num: int, R: int, rng) -> tuple:
    """
    Sample yaw and pitch degrees from a half-circle. 
    Divide the half-circle into eight quadrants, and perform rejection sampling within each quadrant.

    Args:
        density (list): 
            A list of root density proportions per quadrant.
        root_num (int): 
            The number of roots.
        R (int): 
            The radius.
        rng (RNG):
            The random number generator.

    Returns:
        tuple: 
            The yaw and pitch.
    """
    density = [ float(d) for d in density ]
    if len(density) != 8:
        raise Exception("Root density for sampling distribution of roots must contain 8 values.")
    if sum(density) != 1.0:
        raise Exception("Root density for sampling distribution of roots must sum to 1.")
    
    yaw_list = []
    pitch_list = []
    yaw_base = 0

    def __sample_half_circle(quadrant_len: int, pitch_intervals: tuple):
        yaw_samples = []
        while len(yaw_samples) < quadrant_len:
            phi = rng.random(size = 1).item() * np.pi 
            r = np.sqrt(rng.random(size = 1).item()) * R 
            yaw = r * np.cos(phi) * 180 + 180
            if yaw >= yaw_base and yaw <= yaw_base + 90: 
                yaw_samples.append(yaw)
        yaw_list.extend(yaw_samples)

        pitch_samples = []
        lower_pitch, upper_pitch = pitch_intervals
        while len(pitch_samples) < quadrant_len:
            phi = rng.random(size = 1).item() * np.pi 
            r = np.sqrt(rng.random(size = 1).item()) * R 
            pitch = r * np.sin(phi) * 180 
            if pitch >= lower_pitch and pitch <= upper_pitch:
                pitch -= 45
                pitch_samples.append(pitch)
        pitch_list.extend(pitch_samples)

    for i in range(0, len(density), 2):
        upper_quadrant_len = math.ceil(density[i] * root_num)
        lower_quadrant_len = math.ceil(density[i + 1] * root_num)

        if upper_quadrant_len > 0:
            __sample_half_circle(upper_quadrant_len, (90, 180))
        if lower_quadrant_len > 0:
            __sample_half_circle(lower_quadrant_len, (0, 90))

        yaw_base += 90

    yaw = np.array(yaw_list)
    pitch = np.array(pitch_list)
    return yaw, pitch

def sample_half_cone(root_num: tuple, visualise_roots: bool, rng, density: np.ndarray = None) -> tuple:
    """
    Sample yaw, pitch, and roll degrees from a half-circle.

    Args:
        root_num (tuple): 
            The number of roots.
        visualise_roots (bool): 
            Whether to visualise the roots.
        rng (RNG): 
            The random number generator.
        density (list): 
            A list of root density proportions per quadrant.

    Returns:
        tuple: 
            The yaw, pitch, and roll.
    """
    yaw, pitch = sample_half_circle(root_num, visualise_roots, rng, density)
    roll = rng.uniform(50, 75, root_num) # 45 deg => Nearly parallel to parent. 120 deg => Nearly perpendicular to parent          
    return yaw, pitch, roll