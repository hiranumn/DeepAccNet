#!/bin/bash
#SBATCH --job-name=refinement
#SBATCH -p yourpartition
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mem=128g
#SBATCH --time=72:00:00
#SBATCH -o run.log

source $DEEPACCNETPATH/modeling/setup.sh

echo "DeepAccNet on init.pdb"
python $DANPATH/DeepAccNet.py --pdb init.pdb

echo "Initial diversification"
python $SCRIPTPATH/MainDiversification.py  idiv init.npz 

echo "Iterative intensification"
python $SCRIPTPATH/MainIteration.py  idiv/pick.Q.out init.npz ihyb.a

echo "Post-processing"
python $SCRIPTPATH/postprocess.py ihyb.a

exit 0
