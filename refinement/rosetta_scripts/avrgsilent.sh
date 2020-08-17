#!/bin/bash

silent=$1
refpdb=$2
dcut=$3
outprefix=$4

$ROSETTAPATH/source/bin/avrg_silent$ROSETTASUFFIX \
    -in:file:silent $silent -template_pdb $refpdb -cm:similarity_cut $dcut \
    -score:weights ref2015_cart -out:prefix $outprefix \
    -silent_read_through_errors -relax:constrain_relax_to_start_coords \
    > trj.avrg.log
