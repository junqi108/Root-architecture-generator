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
# check more parameters at pages et al., 2014: Calibration and evaluation of ArchiSimple, a simple model of root
# system architecture
class DeVriesParameters:
    """Root system parameters from de Vries et al. (2021)."""
    def __init__(self, species, rng) -> None:
        # Constants for both species
        self.groot: float = 0.0075 # Base rate of Gravitropism 
        self.rzone: float = -1e-4 # No lateral zone in cm
        # varies with species
        self.ibd: Tuple[float, float] = (0.0078, 0.0031) # Inter Branch Distance
        self.dinit: Tuple[float, float] = (0.11, 3.1) # Initial root Diameter in cm for first species, 3.1 for grapevine
        self.angle_avg: Tuple[float, float] = (60, 60) # Average insertion angle of lateral roots
        self.angle_var: Tuple[float, float] = (20, 20) # Variation in the insertion angle of lateral roots
        self.rsp: Tuple[float, float] = (0.5, 0.27) # Ratio between secondary and primary roots
        self.rts: Tuple[float, float] = (0.375, 0.375) # Ratio between tertiary and secondary roots for first species, 0.27 for grapevine
        self.mcp: Tuple[float, float] = (50, 50) # Mechanical constraints; radial degrees/m
        self.sdd: Tuple[float, float] = (0, 0.1) # Standard deviation diameter for RDM
        self.rtd: float = 0.05 if species == 0 else 0.15 # Root tissue density (g/cm3), different for grapevine
                
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
        return self.rng.normal(self.angle_avg[self.species], self.angle_var[self.species])

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
