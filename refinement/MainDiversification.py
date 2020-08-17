import os,sys,glob
import numpy as np

MYPATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTPATH = os.environ.get('SCRIPTPATH')
DANPATH = os.environ.get('DANPATH')
PARALLEL = os.environ.get('GNUPARALLEL')
import utils
from config import CONFIGS
import QScorer

CURR = os.getcwd()

def run_trg(workpath,predf,
            opt=['cons','aggr']):

    # check files/options
    utils.check_files([predf,'init.pdb','t000_.3mers','t000_.9mers'])
    if not os.path.exists('input.fa'): utils.pdb2fa('init.pdb',outfa='input.fa',gap=False)
    if not os.path.exists('disulf.def'): os.system('echo  "1 1" > disulf.def')
    
    extraopt = ''
    if os.path.exists('native.pdb'):
        extraopt += '-native %s/native.pdb'%CURR

    # read in options
    nstruct = CONFIGS['nstruct_div']
    njobs = CONFIGS['njobs_div']
    nproc = CONFIGS['nproc']
        
    os.system('mkdir %s 2>/dev/null'%workpath)
    os.chdir(workpath)
    
    # Restraint (or "cst") generation using estogram2cst script
    if not os.path.exists('cons.cst'):
        print('CMD: python %s/estogram2cst.py %s/%s %s/init.pdb cons -weakcst spline -reference_correction > cons.cstgen.txt'%(SCRIPTPATH,CURR,predf,CURR))
        os.system('python %s/estogram2cst.py %s/%s %s/init.pdb cons -weakcst spline -reference_correction > cons.cstgen.txt'%(SCRIPTPATH,CURR,predf,CURR))

    if not os.path.exists('aggr.cst'):
        aggropt = '-exulr_from_harm -reference_correction -pcore 0.8 0.8 0.9'
        print('CMD: python %s/estogram2cst.py %s/%s %s/init.pdb aggr -weakcst spline %s > aggr.cstgen.txt'%(SCRIPTPATH,CURR,predf,CURR,aggropt))
        os.system('python %s/estogram2cst.py %s/%s %s/init.pdb aggr -weakcst spline %s > aggr.cstgen.txt'%(SCRIPTPATH,CURR,predf,CURR,aggropt))

    # Append input cst if exists
    if os.path.exists('../input.fa.cst'):
        os.system('cat ../input.fa.cst >> cons.fa.cst')
        os.system('cat ../input.fa.cst >> aggr.fa.cst')

    if os.path.exists('../input.cen.cst'):
        os.system('cat ../input.cen.cst >> cons.cst')
        os.system('cat ../input.cen.cst >> aggr.cst')

    ## Put all jobs at 'alljobs.all' and run altogether
    jobs = open('alljobs.all','w')

    sh = '%s/rosetta_scripts/runhybrid.sh'%SCRIPTPATH

    # Setup Aggressive sampling jobs
    if 'aggr' in opt:
        outsilent = 'aggr.out'
        pdbstr = " template1=partial.init.aggr.pdb"
        scriptvars = pdbstr + ' cencst=aggr.cst facst=aggr.fa.cst cst_weight=0.2 cst_fa_weight=0.2 scriptdir=%s'%SCRIPTPATH

        for k in range(njobs):
            prefix = 'aggr.%d'%k
            jobs.write('%s %d %s "%s" %s %s "%s"\n'%(sh,nstruct,'mut.xml',
                                                     scriptvars,
                                                     outsilent,prefix,
                                                     extraopt))
        
    # Setup Conservative sampling jobs
    if 'cons' in opt:
        outsilent = 'cons.out'
        pdbstr = "' template1=partial.init.cons.pdb'"
        scriptvars = pdbstr + ' cencst=cons.cst facst=cons.fa.cst cst_weight=0.2 cst_fa_weight=1.0 scriptdir=%s'%SCRIPTPATH

        for k in range(njobs):
            prefix = 'cons.%d'%k
            jobs.write('%s %d %s "%s" %s %s "%s"\n'%(sh,nstruct,'mut.xml',
                                                     scriptvars,
                                                     outsilent,prefix,
                                                     extraopt))
    jobs.close()

    # Launch jobs through gnu parallel
    print('CMD: %s -j %d :::: alljobs.all'%(PARALLEL,nproc))
    os.system('%s -j %d :::: alljobs.all'%(PARALLEL,nproc))

    # Check the number of files produced
    ncons = len(os.popen('grep ^SCORE cons.out').readlines())
    naggr = len(os.popen('grep ^SCORE aggr.out').readlines())
    n_to_sample_per_opt = int(nstruct*njobs*0.9)
    if ncons < n_to_sample_per_opt or naggr < n_to_sample_per_opt:
        sys.exit("Insufficient decoys generated: %d/%d in cons/aggr.out, terminate!"%(ncons,naggr))

    # clear logs if sampled numbers are okay
    os.system('rm cons*log aggr*log')

    # post process -- pick 50 from generated decoys
    Q = np.mean(np.load(CURR+'/'+predf)['lddt'])
    f = max(0.0,(Q-0.6)/0.4)
    dcut = max(0.2,(1.0-f)*0.3)
        
    print('CMD: %s/rosetta_scripts/pick_from_div.sh %6.2f'%(SCRIPTPATH,dcut))
    os.system('%s/rosetta_scripts/pick_from_div.sh %6.2f'%(SCRIPTPATH,dcut))

    QScorer.main('pick.out','pick.Q.out')
    
    os.chdir(CURR)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("USAGE: python MainDiversification.py [workpath] [DAN pred file as .npz]")
    workpath = sys.argv[1]
    predf = sys.argv[2] #DAN output npz file
    opt = ['cons','aggr']
    
    run_trg(workpath,predf,opt=opt)
    
