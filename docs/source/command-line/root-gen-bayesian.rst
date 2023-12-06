Bayesian Root Generator 
=================================================

=================
Usage
=================

This script will construct a Bayesian statistical model using the Approximate Bayesian Computation (ABC) method. More specifically, the posterior of the Bayesian model will be evaluated by approximating the likelihood of the statistical model.
 
To accomplish this, the model will generate synthetic root data with respect to some summary statistics. Data that exceed a dissimilarity threshold between observed and generated summary statistics will be discarded, while those within the threshold will be retained.

`The PyMC probablistic programming language has been used to construct this model. <https://www.pymc.io/welcome.html>`_ 

The *draws* parameter is extremely important. This determines the number of samples to draw from the posterior. You will typically want somewhere between 100-1000 draws per chain.

The *chains* parameter determines the number of independent Markov chains to run.

The default dissimilarity *distance* is the Euclidean distance. 

The current priors specified for each parameter are uninformative (i.e. uniformly distributed). It is possible to specify the lower and upper bounds for each prior.

Generally speaking, the wider the intervals of each prior, the more difficult the sampling process.

However, using very constrained intervals will lead to very similar looking root systems - there will be a lack of diversity in the resulting synthetic root systems.

Please review the documentation below for supported model parameters.

=================
Arguments
=================
.. argparse::
   :filename: ../root_gen_bayesian.py
   :func: get_parser
   :prog: root_gen_bayesian
