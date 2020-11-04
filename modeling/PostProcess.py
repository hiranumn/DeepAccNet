import os,sys,glob,copy
import numpy as np

SCRIPTPATH = os.environ.get("SCRIPTPATH")
ROSETTAPATH = os.environ.get("ROSETTAPATH")
DLPATH = os.environ.get("DANPATH")
CURR = os.getcwd()
import utils

def avrg_trj(dcut=0.2):
    if os.path.exists('Qsel.avrg.relaxed.pdb'): return
    print("Running structural averaging around Qsel.pdb, dcut=%.2f..."%(dcut))

    os.system('cat iter_*/gen.retag.out > gen.total.out')
    cmd = '%s/rosetta_scripts/avrgsilent.sh %s %s %.3f %s'%(SCRIPTPATH,"gen.total.out","Qsel.pdb",dcut,"Qsel.avrg")

    print("CMD: ", cmd)
    os.system(cmd)
    os.system('cp Qsel.avrg.relaxed.pdb %s'%(CURR))
    # outcome: Qsel.avrg.relaxed.pdb

def main(ihyb_path):
    os.chdir(ihyb_path)
    its = [int(l.split('/')[0].split('_')[-1]) for l in glob.glob('iter_*/ref.out')]
    its.sort()
    fit = its[-1]
    print("Working on %s, final iteration iter_%d detected"%(ihyb_path,fit))

    l = utils.scparse('iter_%d/ref.out'%fit,['Q'])[1]
    maxQ = float(l.split()[-1])
        
    # Get representative model -- since Qpool been used, model0 always has the best Q
    if not os.path.exists('Qsel.pdb'):
        os.system('ln -s iter_%d/iter%d.%d.pdb ./Qsel.pdb'%(fit,fit,0))
    print("%s Qsel: iter%d.%d.pdb, Q=%.3f"%(ihyb_path,fit,0,maxQ))

    # Average trajectory around the representative model followed by relaxation
    avrg_trj()
    os.chdir(CURR)

if __name__ == "__main__":
    ihyb_path = sys.argv[1]
    main(ihyb_path)
