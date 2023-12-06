"""
Root System Library

A library of configuration tools for synthetic root system generation.
"""

##########################################################################################################
### Imports
##########################################################################################################

# External
import argparse
import numpy as np
import os
import yaml

from typing import Any

##########################################################################################################
### Library
##########################################################################################################

class Config:
    """General configuration class"""
    def __init__(self):
        self.config = {}

    def get(self, key: str) -> Any:
        """
        Get a configuration value.

        Parameters
        --------------
        key: str
            The configuration value key.

        Returns
        ---------
        definition : Any
            The configuration value.
        """
        return self.config.get(key)

    def get_as(self, key: str, v_type: type) -> Any:
        """
        Get a configuration value, and cast it to data type.

        Parameters
        --------------
        key: str
            The configuration item key.
        v_type: type
            The data type.

        Returns
        ---------
        definition : Any
            The casted configuration item.
        """
        return v_type(self.config.get(key))

    def split(self, key: str, split_str: str):
        return self.get_as(key, str).split(split_str)

    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
        return self

    def from_env(self):
        """
        Add configuration values from environment variables.

        Returns
        ---------
        self : Config
            The configuration object.
        """
        environment_vars = dict(os.environ)
        return self.extract_key_values(environment_vars)

    def from_yaml(self, path: str):
        """
        Add configuration values from a YAML file.

        Parameters
        --------------
        path: str
            The YAML file path.

        Returns
        ---------
        self : Config
            The configuration object.
        """
        with open(path) as f:
            options: dict = yaml.safe_load(f)
            if options is not None:
                self.extract_key_values(options)
        return self

    def to_yaml(self, outfile: str, default_flow_style: bool = False):
        """
        Export the configuration to a YAML file.

        Parameters
        --------------
        outfile: str
            The outputted YAML file path.
        default_flow_style: bool
            The YAML format style.

        Returns
        ---------
        self : Config
            The configuration object.
        """
        with open(outfile, 'w') as f:
            yaml.dump(self.config, f, default_flow_style = default_flow_style)
        return self

    def from_parser(self, parser: argparse.ArgumentParser):
        """
        Add configuration values from an argument parser.

        Parameters
        --------------
        parser: ArgumentParser
            The argument parser.

        Returns
        ---------
        self : Config
            The configuration object.
        """
        args = parser.parse_args()
        arg_dict = vars(args)
        return self.extract_key_values(arg_dict)

    def extract_key_values(self, options: dict):
        """
        Extract key value pairs from a dictionary, and add them to the current configuration.

        Parameters
        --------------
        options: dictionary
            The dictionary.

        Returns
        ---------
        self : Config
            The configuration object.
        """        
        for k, v in options.items():
            self.config[k] = v
        return self

def add_argument(parser: argparse.ArgumentParser, name: str, default, arg_help: str, 
    type: type = int, choices: list = None) -> None:
    """
    Add argument to the argument parser.

    Parameters
    --------------
    parser: parser
        The argument parser.
    name: str
        The argument name.
    default: any
        The default value.
    arg_help: str
        The argument help text.
    type: type
        The argument type.
    choices: list
        Choices for multiple choice options.
    """
    parser.add_argument(
        name,
        default = default,
        help = f"{arg_help}. Defaults to '{default}'.",
        type = type,
        choices = choices
    )

def construct_interval(config: Config, k1: str, k2: str) -> np.ndarray:
    """
    Construct an interval using two configuration values.

    Parameters
    --------------
    config : Config
        The configuration object.
    k1: str
        The key of the lower bound.
    k2: str
        The key of the upper bound.

    Returns
    ---------
    interval : (2,)
        The interval.
    """ 
    return np.array([config.get(k1), config.get(k2)])


    