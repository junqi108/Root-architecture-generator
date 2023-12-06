Generate summary statistics
=================================================

=================
Usage
=================

This script will generate summary statistics from a file of observed values.

The *root_stats* parameter is a comma-delimited list of root statistics which are mapped to a column by name. 

For example, try running the following command: `generate_summary_stats.py --root_stats depth_cum=soil_depth_m,depth_cum=radial_distance_from_stem_m` 

The *obs_file* file should ideally be real field data that you wish to convert to a compatible format for the root simulator.

The *sblock_size* parameter is used for determining the size of each soil block when dividing the soil system into a grid.

=================
Arguments
=================
.. argparse::
   :filename: ../generate_summary_stats.py
   :func: get_parser
   :prog: generate_summary_stats