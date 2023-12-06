Root Generator 
=================================================

=================
Usage
=================

This script is used to generate a synthetic root system.

Several root parameters have been taken from the paper: Mycorrhizal associations change root functionality: a 3D modelling study on competitive interactions between plants for light and nutrients (de Vries et al., 2021).

By default, synthetic data are written to *data/root_sim/root_sim.csv.*

The *nplants* parameter specifies the number of plants to generate root systems for.

The *morder* parameter determines the maximum root order.

The *froot_threshold* parameter determines the root diameter threshold for classifying a root as either a structural or fine root.

=================
Arguments
=================
.. argparse::
   :filename: ../root_gen.py
   :func: get_parser
   :prog: root_gen