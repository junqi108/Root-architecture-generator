"""
Root System Library

A library of model parameters for synthetic root system generation.
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

# de Vries et al. (2021) Mycorrhizal associations change root functionality...
class DeVriesParameters:
    """Root system parameters from de Vries et al. (2021)."""
    def __init__(self, species, rng) -> None:
        if species == 0:
            # Constants
            self.ibd: Tuple[float] = (0.0078, 0.0078) # Inter Branch Distance
            self.dinit: Tuple[float] = (0.11, 0.11) # Initial root Diameter in cm
            self.groot: float = 0.0075 # Base rate of Gravitropism 
            self.rzone: float = -1e-4 #-0.001 # No lateral zone in cm
            self.angle_avg: float = 60 # average insertion angle of lateral roots
            self.angle_var: float = 20 # variation in the insertion angle of lateral roots
            self.rsp: Tuple[float] = (0.7, 0.5) # Ratio between secondary and primary roots
            self.rts: Tuple[float] = (0.375, 0.375) # Ratio between tertiary and secondary roots
            self.avg_root_r: float = 0.00105 # average root radius
            self.mcp: Tuple[float] = (50, 50) # random root movement based on mechanical staticraints; radial degrees/m
            self.sdd: Tuple[float] = (0, 0) # Standard deviation diameter for RDM
            self.rtd: float = 0.05 # Root tissue density (g/cm3)

        elif species == 1:
            # Constants grapvine
            self.ibd: Tuple[float] = (0.0078, 0.0078) # Inter Branch Distance
            self.dinit: Tuple[float] = (3.1, 3.1) # Initial root Diameter in cm
            self.groot: float = 0.0075 # Base rate of Gravitropism 
            self.rzone: float = -1e-4 #-0.001 # No lateral zone in cm
            self.angle_avg: float = 60 # average insertion angle of lateral roots
            self.angle_var: float = 20 # variation in the insertion angle of lateral roots
            self.rsp: Tuple[float] = (0.7, 0.5) # Ratio between secondary and primary roots
            self.rts: Tuple[float] = (0.375, 0.375) # Ratio between tertiary and secondary roots
            self.avg_root_r: float = 0.00105 # average root radius
            self.mcp: Tuple[float] = (50, 50) # random root movement based on mechanical staticraints; radial degrees/m
            self.sdd: Tuple[float] = (0, 0) # Standard deviation diameter for RDM
            self.rtd: float = 0.05 # Root tissue density (g/cm3)
        else:
            # Handle unexpected species value
            raise ValueError(f"Unknown species: {species}")
        
        self.species: int = species
        self.rng: np.random.Generator = rng

    def get_gravitropism_factor(self) -> float:
        """
        Get the gravitropism factor for root development.

        Returns
        ---------
        gravitropism factor : float  
            The gravitropism factor
        """
        return self.groot * 1e3

    def get_gravitropism(self, diameter: float) -> float:
        """
        Calculate gravitropism as a function of root diameter.

        Parameters
        --------------
        diameter: float
            The root diameter.

        Returns
        ---------
        gravitropism : float  
            The gravitropism level.
        """
        return self.groot * (diameter * 1e3)

    def get_angle(self) -> float:
        """
        Sample the lateral root angle from a normal distribution.

        Returns
        ---------
        angle : float  
            The root angle.
        """
        return self.rng.normal(self.angle_avg, self.angle_var)

    def get_rdm(self) -> float: 
        """
        The diameter reduction factor of lateral roots.

        Returns
        ---------
        rdm : float  
            The diameter reduction factor
        """
        return self.rsp[self.species]

    def get_dinit(self) -> float:
        """
        The initial root diameter.

        Returns
        ---------
        diameter : float  
            The initial diameter.
        """
        return self.dinit[self.species]

    def get_mcp(self) -> float:
        """
        The proportion of mycorrhizal colonisation within the root system.

        Returns
        ---------
        mcp : float  
            The proportion of mycorrhizal colonisation.
        """
        return self.mcp[self.species]

    def get_ibd(self) -> float:
        """
        The interbranching distance between lateral roots, with the highest root order having longer segments to improve computation time

        Returns
        ---------
        ibd : float  
            The interbranching distance between lateral roots.
        """
        return self.ibd[self.species]

    def get_next_diameter(self, diameter) -> float:
        """
        Return the diameter of subsequent root segments, reduced by a scaling factor.

        Parameters
        --------------
        diameter: float
            The root diameter.

        Returns
        ---------
        reduced : float  
            The reduced diameter.
        """
        diameter_base = self.get_rdm() * diameter
        diameter_variance = self.sdd
        if diameter_variance > 0:
            reduced_diameter =  self.rng.normal(diameter_base, diameter_base * diameter_variance)
        else:
            reduced_diameter = diameter_base
        return reduced_diameter
