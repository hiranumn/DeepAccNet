#!/bin/bash

# Script picking 50 structs from two silent files [cons.out,aggr.out] with quota [12,38]
# with struct tags "iter0" (e.g. iter0.0, iter0.1, ... iter0.49)
# dcut is 1-Sscore scale; recommended range [0.2,0.6]: 0.6 gives very diverse results, 0.2 very tight results

dcut=$1
extra=$2

rm pick.out 2>/dev/null # just in case to avoid appending

$ROSETTAPATH/source/bin/iterhybrid_selector$ROSETTASUFFIX \
 -silent_read_through_errors \
 -similarity_cut $dcut \
 -in:file:template_pdb ../init.pdb \
 -out:file:silent pick.out \
 -in:file:silent_struct_type binary -silent_read_through_errors \
 -mute core.conformation \
 -out:nstruct 50 \
 -in:file:silent cons.out aggr.out \
 -cm::quota_per_silent 12 38 \
 -out:prefix iter0 \
 $extra \
 > pick.log

