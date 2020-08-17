#!/bin/bash

n=$1
script=$2
script_vars=$3
outsilent=$4
runprefix=$5
extra=$6

echo $ROSETTAPATH/source/bin/rosetta_scripts$ROSETTASUFFIX
$ROSETTAPATH/source/bin/rosetta_scripts$ROSETTASUFFIX \
    @$SCRIPTPATH/rosetta_scripts/flags \
    -parser:protocol $SCRIPTPATH/rosetta_scripts/$script \
    -parser:script_vars $script_vars \
    -nstruct $n \
    -frag_weight_aligned 0.2 \
    -out:file:silent $outsilent \
    -out:prefix $runprefix \
    $extra \
    > $runprefix.log
