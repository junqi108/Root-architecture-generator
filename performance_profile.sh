#!/bin/bash

##########################################################################################################
### Constants
##########################################################################################################

DATA_DIR=data/root_sim
OUTPUT=${DATA_DIR}/root_gen.pstats
IMG=${DATA_DIR}/root_gen.png

##########################################################################################################
### Main
##########################################################################################################

### Install gprof2dot first
### pip install gprof2dot --user

python -m cProfile -o $OUTPUT root_gen.py "${@}"
echo "Profile written to ${OUTPUT}"
python -m gprof2dot -f pstats $OUTPUT | dot -Tpng -o $IMG # && eog $IMG
echo "Visualisation written to ${IMG}"
