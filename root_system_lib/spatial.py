"""
Root System Library

A library of methods relating to spatial properties of synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
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
