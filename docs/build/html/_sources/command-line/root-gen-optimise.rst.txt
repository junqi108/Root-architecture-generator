Optimisation Root Generator 
=================================================

=================
Usage
=================

This script will optimise the generation of a synthetic root system with respect to some summary statistics. 

More specifically, it will aim to minimise the dissimilarity between the observed and generated summary statistics.

`The Optuna optimisation library has been used to perform this optimisation procedure. <https://optuna.org/>`_ 

The *n_trials* parameter is extremely important. This determines the number of iterations to perform optimisation. 

Depending on the difficulty of the optimisation task, hundreds to thousands of iterations may be required. 

Try to execute the optimisation procedure in blocks of *500* trials, and see if the dissimilarity metric for the observed and generated summary statistics continues to decrease. 

The default dissimilarity *distance* is the Euclidean distance. 

The *load_optimiser* parameter can be used to load an existing optimiser object and its respective trial history, allowing you to resume the optimisation procedure.

The Optuna library will sample from a uniform distribution for each model parameter. It is possible to specify the lower and upper bounds for each model parameter.

Generally speaking, the wider the intervals of each model parameter, the more difficult the optimisation task.

However, using very constrained intervals will lead to very similar looking root systems - there will be a lack of diversity in the resulting synthetic root systems.

Please review the documentation below for supported model parameters.

=================
Arguments
=================
.. argparse::
   :filename: ../root_gen_optimise.py
   :func: get_parser
   :prog: root_gen_optimise

