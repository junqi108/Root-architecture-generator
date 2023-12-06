"""
Root System Library

A library for computing dissimilarities between data from synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import numpy as np

from typing import Tuple

##########################################################################################################
### Library
##########################################################################################################

def distance_euclidean(e: float, obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes Euclidean distance between data.

    Parameters
    --------------
    e: float
        Epsilon parameter for compatability with PyMC3. 
    obs: array (float)  
        Observed data.
    sim: array (float)
        Simulated data.

    Returns
    ---------
    distance : float | array (float)
        The computed euclidean distances.
    """
    dist = np.linalg.norm(obs - sim)
    return dist * e

ROOT_DISTANCES = {
    "euclidean": distance_euclidean
}
 