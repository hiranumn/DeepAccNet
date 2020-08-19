#!/bin/bash

# tensorflow 
source activate /software/conda/envs/tensorflow

# Rosetta
export ROSETTAPATH=/software/rosetta/latest/main
export ROSETTASUFFIX=''

# pyrosetta3 setup
source /software/pyrosetta3.6/setup.sh

# DeepAccNet
export DANPATH=/home/hpark/programs/DeepAccNet.git

# Path to this script
export SCRIPTPATH=$DANPATH/modeling

# GNU parallel
export GNUPARALLEL=/usr/bin/parallel
