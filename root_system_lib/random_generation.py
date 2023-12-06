"""
Root System Library

A library of tools for random number generation for synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import numpy as np

from numpy.random import default_rng

##########################################################################################################
### Library
##########################################################################################################

def get_rng(random_seed: int = None) -> np.random.Generator:
    """
    Return a new random number generator instance.

    Parameters
    --------------
    random_seed: int
        The seed value.

    Returns
    ---------
    generator: Generator
        The random number generator.
    """

    return default_rng(random_seed)