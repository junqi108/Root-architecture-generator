Plot root
=================================================

=================
Usage
=================

This script will allow you to visualise generated root systems.

The *plant* parameter will allow you to specify individual plants to visualise.

The *soil_grid* parameter will render grid of soil blocks within the root system. This grid is used for certain summary statistics.

The *detailed_vis* parameter is used to specify whether detailed root information is included in the plot. The more information that is included. The longer it takes to render the plot.

=================
Arguments
=================
.. argparse::
   :filename: ../plot_root.py
   :func: get_parser
   :prog: plot_root