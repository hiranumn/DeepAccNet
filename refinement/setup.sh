#!/bin/bash

# tensorflow 
source /software/conda/bin/activate /software/conda/envs/tensorflow
#source activate /software/conda/envs/tensorflow

# pyrosetta3 setup
source /software/pyrosetta3.6/setup.sh

# Path to this script directory
export SCRIPTPATH=/projects/casp/RefinementScripts/simple/

# Rosetta
export ROSETTAPATH=$SCRIPTPATH/Rosetta
export ROSETTASUFFIX=''
#export ROSETTASUFFIX='.linuxgccrelease'

# DeepAccNet
export DANPATH=$SCRIPTPATH/DeepAccNet

# GNU parallel
export GNUPARALLEL=/usr/bin/parallel
