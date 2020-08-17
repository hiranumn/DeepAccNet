#!/usr/bin/python

import os,copy,time,glob,random,sys
import numpy as np

MYPATH = os.path.dirname(os.path.realpath(__file__))
SCRIPTPATH = os.environ['SCRIPTPATH']
DANPATH = os.environ['DANPATH']
ROSETTAPATH = os.environ['ROSETTAPATH']
ROSETTASUFFIX = os.environ['ROSETTASUFFIX']
PARALLEL = os.environ['GNUPARALLEL']
sys.path.insert(0,SCRIPTPATH)

import Qpool as Qpool
import recomb.find_partners 
import utils

## Parsing arguments
def parse_args(argv):
    import argparse
    import config

    # read default first -- see descriptions in the file for help
    parser = argparse.ArgumentParser(description="")
    for key in config.CONFIGS:
        val = config.CONFIGS[key]
        if isinstance(val,bool):
            parser.add_argument("-"+key, default=val, action="store_true")
        else:
            parser.add_argument("-"+key, type=type(val), default=val)
    opt = parser.parse_args(argv)

    # setup dependent options (e.g. symmetry)
    opt.curr = os.getcwd()
    opt.niter_dcut_decrease = int(0.5*opt.niter)
    opt.maxiter_reset = 50*3/opt.nseed #==15
    
    if opt.symmetric:
        opt.xml_cross1_suffix = opt.xml_cross1_suffix[:-4]+'.symm.xml'
        opt.xml_cross2_suffix = opt.xml_cross2_suffix[:-4]+'.symm.xml'
        opt.xml_mut_suffix    = opt.xml_mut_suffix[:-4]+'.symm.xml'

    if opt.debug: opt.ngen_per_job=1

    # check consistency
    if opt.symmetric and opt.symmdef == '':
        sys.exit("-symmdef [file] should be defined with '-symmetric option!")
    if opt.pool_update not in ['standard','Q','Qlocal']:
        sys.exit("Error: -pool_update [standard/Q/Qlocal]")
    
    # report all options
    print("Verbose options:")
    for key in opt.__dict__:
        print("OPTION", key, getattr(opt,key))
    
    return opt

class RunInfo:
    def __init__(self,argv):
        self.it = 0
        self.curr = os.getcwd()
        self.phase = 0
        self.phase_update_it = 0
        self.seldata = SelectionData(0,'',0)
        self.opt = parse_args(argv)
        self.dcut0_dynamic = 0.0

    def get_script(self,mode,phase,itstr,iseed,script_vars_add,cstprefix,autocst=False):
        scriptdir = MYPATH
        script_mut = self.opt.xml_mut_suffix
        script_cross = self.opt.xml_cross1_suffix

        if autocst:
            cencst = 'AUTO'
        else:
            cencst = '%s.cst'%cstprefix
        facst = '%s.fa.cst'%cstprefix

        script_vars = " scriptdir=%s/ "%scriptdir +\
            " cencst=%s cst_weight=%s facst=%s cst_fa_weight=%s"%(cencst,self.opt.cstweight,facst,self.opt.cst_fa_weight) +\
            script_vars_add
        script_vars = '"%s"'%script_vars

        extra = ''
        if self.opt.native != '':
            extra += ' -in:file:native %s'%self.opt.native
        if self.opt.symmetric:
            extra += ' -symmetry_definition %s'%self.opt.symmdef
        extra = '"'+extra+'"'

        outsilent = 'gen.out'
        workpath = "%s/iter_%d"%(self.curr,self.it)
        if mode[:5] == 'cross':
            form = '%s/rosetta_scripts/runhybrid.sh 1 %s %s %s %s %s'
            prefix = '%s_from%d_%s_'%(itstr,iseed,mode)
            return form%(scriptdir,script_cross,script_vars,outsilent,prefix,extra)
        elif mode == 'mut':
            form = '%s/rosetta_scripts/runhybrid.sh 1 %s %s %s %s %s'
            prefix = '%s_from%d_mut_'%(itstr,iseed)
            return form%(scriptdir,script_mut,script_vars,outsilent,prefix,extra)

    def nhybrid(self):
        return self.opt.nperseed*self.opt.nseed

    def get_ngen_per_job(self,phase):
        if phase == 0:
            return int(self.opt.mulfactor_phase0*self.opt.ngen_per_job)
        else:
            return self.opt.ngen_per_job

    def ngen(self,phase):
        if phase == 0:
            return int(self.opt.mulfactor_phase0*self.opt.nseed*self.opt.nperseed*self.opt.ngen_per_job)
        else:
            return self.opt.nseed*self.opt.nperseed*self.opt.ngen_per_job
        
    def update_phase(self,phasefile):
        if os.path.exists(phasefile):
            words = open(phasefile).read().split()
            self.phase = int(words[0])
            self.phase_update_it = int(words[1])

    def update_dcut(self):
        if self.it == 0:
            if not os.path.exists(self.curr+'/dcut0.txt'):
                self.dcut0_dynamic = self.get_dcut0_dynamic(glob.glob('%s/iter_%d/iter0.*pdb'%(self.curr,self.it)),
                                                            '%s/iter_%d/iter0.0.pdb'%(self.curr,self.it))
                out = open(self.curr+'/dcut0.txt','w')
                out.write(str(self.dcut0_dynamic)+'\n')
                out.close()
            else:
                self.dcut0_dynamic = float(open(self.curr+'/dcut0.txt').read())
            self.dcut = copy.copy(self.dcut0_dynamic)
        else:
            if self.dcut0_dynamic == 0.0:
                self.dcut0_dynamic = float(open(self.curr+'/dcut0.txt').read())

            if self.it >= self.opt.niter_dcut_decrease:
                self.dcut = self.dcut0_dynamic*self.opt.dcut_min_scale
            else:
                f = 1.0 - self.opt.dcut_min_scale*float(self.it)/self.opt.niter_dcut_decrease
                self.dcut = self.dcut0_dynamic*f

    def get_dcut0_dynamic(self,pdbs,refpdb):
        # f = 1.0 until iha = 40, decrease as gets easier (iha = 60 gives f = 0.5)
        f = (self.opt.iha-40.0)/40.0
        if f < 0.0: f = 0.0
        return (1.0-f)*self.opt.dcut0 

    def nmut(self):
        if self.phase in self.opt.nmutperseed: 
            return self.opt.nmutperseed[self.phase]
        else:
            return 0

    def simlimit(self):
        if self.opt.simlimit_base < self.dcut:
            simlimit = self.opt.simlimit_base
        else:
            simlimit = self.dcut
        return simlimit

    def is_recomb_iter(self,it):
        if self.opt.recomb_iter <= 0: return False
        if it%self.opt.recomb_iter == 0:
            return True
        else:
            return False

    def is_reset(self,n0):
        if n0 <= self.opt.n0_to_reset or \
                self.it-self.phase_update_it >= self.opt.maxiter_reset:
            return True
        else:
            return False

class SelectionData:
    def __init__(self,it,refout,nseeds,scoretype='score',
                 scoresign=1):
        self.seeds = []
        self.tags = []
        self.poolids = {}
        self.refout = refout
        self.scoretype = scoretype
        self.scoresign = scoresign
        if refout != '':
            self.get_seeds(it,nseeds)

    # Pick seeds on least-used ones
    def get_seeds(self,it,nseed):
        sortable = []
        self.nuse = {}
        # Tag score nuse

        read_nuse = True
        self.poolids = {}
        for l in open(self.refout):
            if not l.startswith('SCORE:'): continue
            words = l[:-1].split()

            if 'description' in l:
                k_score = words.index(self.scoretype)
                k_pool = words.index('poolid')
                if 'nuse' in words:
                    k_nuse = words.index('nuse')
                else:
                    read_nuse = False
                continue
            else:
                tag = words[-1]
                if tag.endswith('.pdb'): tag = tag.replace('.pdb','')

                score = float(words[k_score])
                if self.scoresign < 0: score *= -1.0
                if read_nuse:
                    try:
                        nuse = int(float(words[k_nuse]))
                    except:
                        nuse = 0
                else:
                    nuse = 0
                self.nuse[tag] = nuse
                self.poolids[tag] = int(float(words[k_pool]))
                sortable.append([nuse,score,tag])
        sortable.sort()
        self.seeds = [comp[2] for comp in sortable[:nseed]]

        self.seedids = {}
        for seed in self.seeds:
            self.nuse[seed] += 1

def add_parents(infile,outfile):
    cont = []
    refcont = utils.all_add_score(infile, 'parent', 0)

    for l in refcont:
        if l.startswith('SCORE'):
            words = l[:-1].split()
            if 'description' in l:
                k_score = words.index('score')
                k_parent = words.index('parent')
                newl = l
            else:
                tag = words[-1]
                parent = int(tag.split('_from')[-1].split('_')[0])
                part1 = words[1:k_parent]
                part2 = words[k_parent+1:]
                newl = 'SCORE:'+' %10s'*len(part1)%tuple(part1)+' %10d'%parent+' %10s'*len(part2)%tuple(part2)+'\n'
            cont.append(newl)
        else:
            cont.append(l)
    out = open(outfile,'w')
    out.writelines(cont)
    out.close()

def DAN_and_gen_restraints(tags_score, tags_cst,
                           outpath='pred', 
                           extra_args='',
                           generate_cst=True,
                           nproc=10
                           ): #null_cst is for alternating cst on/off; unused

    # keep producing DL for book keeping
    if not os.path.exists(outpath) or \
       len(glob.glob('%s/*.?.npz'%outpath)+glob.glob('%s/*.??.npz'%outpath)) < len(tags_score):
        if not os.path.exists(outpath): os.mkdir(outpath)
        for tag in tags_score:
            os.system('cp %s.pdb %s/'%(tag,outpath))
        os.chdir(outpath)
        time0 = time.time()
        print( "Running DL predictions for", " ".join(tags_score) )
        os.system('python %s/ErrorPredictor.py -p %d ./ 1> DANmsa.log'%(DANPATH,nproc))
        time1 = time.time()
        print( " -- DANmsa took %.1f seconds."%(time1-time0))
        os.chdir('..')

    if not generate_cst: return
    
    for tag in tags_cst:
        #make sure npz file exists
        if not os.path.exists('%s/%s.npz'%(outpath,tag)):
            sys.exit('DL result file %s/%s.npz does not exist!'%(outpath,tag))

        os.system('python %s/estogram2cst.py %s/%s.npz %s.pdb %s -weakcst bounded %s > %s.cstgen.txt'%(MYPATH,outpath,tag,tag,tag,extra_args,tag))

        # if input cst provided
        if os.path.exists('../input.cen.cst'): 
            os.system('cat ../input.cen.cst >> %s.cst'%tag)
        if os.path.exists('../input.fa.cst'): 
            os.system('cat ../input.fa.cst >> %s.fa.cst'%tag)

# Use DAN to pick seeds
def prepick(runinfo):
    # 1st priority: nuse
    # 2nd priority: "scoretype" (e.g. Rosetta)

    (scoretype,scoresign) = ("score",1)
    if runinfo.opt.pool_update.startswith('Q'):
        (scoretype,scoresign) = ('Q',-1)
    
    seldata = SelectionData(runinfo.it,'ref.out',runinfo.opt.nseed,
                            scoretype=scoretype,
                            scoresign=scoresign)

    seldata.tags = [l.replace('.pdb','') for l in glob.glob('iter%d*.pdb'%runinfo.it)]

    if runinfo.it > 0 and os.path.exists('../iter_%d/PHASE'%(runinfo.it-1)):
        words = open('../iter_%d/PHASE'%(runinfo.it-1)).read()[:-1].split()
        runinfo.phase = int(words[0])
        runinfo.phase_update_it = int(words[1])
    runinfo.seldata = seldata #update

    # run DAN on all tags if doing DAN-aware recombination
    if runinfo.opt.recomb == 'random':
        tags_score = seldata.seeds
    else: #recombX
        tags_score = seldata.tags #all
    
    # composite of DLnetwork & estogram2cst
    DAN_and_gen_restraints(tags_score,
                           seldata.seeds,
                           outpath="pred",
                           extra_args=runinfo.opt.e2cst_args,
                           nproc=min(len(tags_score),runinfo.opt.nproc))

def postpick(runinfo):
    seldata = runinfo.seldata

    # hack to put parent info as a silent column
    if os.path.exists('gen.out') and not os.path.exists('gen.retag.out'):
        add_parents('gen.out','gen.retag.out')
        os.system('rm gen.out')

    if not os.path.exists('sel.out'):
        seedstr = ' '.join(['%d'%seldata.poolids[seed] for seed in seldata.seeds])
        simlimit = runinfo.simlimit()

        class EmptyClass:
            def __init__(self):
                pass
            
        local_opt = EmptyClass()
        local_opt.refpdb = '%s/init.pdb'%(runinfo.opt.curr)
        local_opt.silent_ref = 'ref.out'
        local_opt.silent_gen = 'gen.retag.out'
        local_opt.silent_out = 'sel.out'
        local_opt.seeds = ['%d'%seldata.poolids[seed] for seed in seldata.seeds]
        local_opt.simlimit = simlimit
        local_opt.dcut = runinfo.dcut
        local_opt.outprefix = "iter%d"%(runinfo.it+1)
        local_opt.mode = runinfo.opt.pool_update #TODO
        local_opt.iha_cut = runinfo.opt.iha
        if runinfo.opt.iha > 1: local_opt.iha_cut *= 0.01
        local_opt.add_init_penalty = True
        local_opt.iha_penalty_slope = 10.0
        local_opt.preserve_nuse_sim = True
        local_opt.min_reset_unused = 3
        local_opt.filterscore = "score"
        local_opt.verbose = False
        sellog=open("sel.log",'w')
        #sellog=sys.stdout
        print("Call Qpool...")
        Qpool.main(local_opt,out=sellog)
        sellog.close()
            
    # Reset nuse
    reset = False
    if runinfo.phase == 2:
        reset = True
    else:
        cont = utils.scparse('sel.out',['nuse'])
        nuses = [int(float(l[:-1].split()[1])) for l in cont[1:]]
        n0 = nuses.count(0)
        reset = runinfo.is_reset(n0)

    if reset:
        runinfo.phase += 1
        cont = utils.reset_score('sel.out','nuse','0.000')
        out = open('sel.out','w')
        out.writelines(cont)
        out.close()
        runinfo.phase_update_it = runinfo.it

    os.system('echo "%d %d"> PHASE'%(runinfo.phase,runinfo.phase_update_it))
    os.system('touch DONE')

def gen_iter(runinfo):
    it = runinfo.it

    currdir = '%s/iter_%d'%(runinfo.curr,it)
    if not os.path.exists(currdir): os.mkdir(currdir)

    if it > 0:
        os.chdir(currdir)
        prvdir = '%s/iter_%d'%(runinfo.curr,it-1)
        os.system('ln -s %s/sel.out ./ref.out 2>/dev/null'%(prvdir))
        os.system('rm %s/*pdb 2>/dev/null'%(prvdir))
        os.chdir(runinfo.curr)

def make_combination(runinfo):
    n = runinfo.nhybrid()
    nper = runinfo.opt.ntempl_per_job

    comb = []
    seeds = runinfo.seldata.seeds
    seed_idx = [int(seed.split('.')[-1]) for seed in runinfo.seldata.seeds]
    tags = runinfo.seldata.tags

    # this is NOT recomb_iter but recomb-suggestion-on-regular-hybiter 
    # turn this off for now
    cross_from_DAN = {}
    if runinfo.opt.recomb != 'random':
        print("seeds?", seed_idx)
        cross_from_DAN = recomb.find_partners.main(infolder='pred',
                                                   out=open('recomb1D.txt','w'),
                                                   seeds=seed_idx)
    
    for seed in seeds:
        nonseeds = []
        n = len(nonseeds)
        for tag in tags:
            if tag not in seeds: nonseeds.append(tag)

        #let's take shuffle for simplicity...
        #mutation
        nmut = runinfo.nmut()
        ncross = runinfo.opt.nperseed-nmut
        for k in range(nmut):
            comb.append([seed])

        #crossover
        for k in range(ncross):
            comb_i = [seed]
            if seed in cross_from_DAN and k%2 == 0: #rest half as random
                comb_i += cross_from_DAN[seed]
            else: #random 
                random.shuffle(nonseeds)
                for i in range(runinfo.opt.ntempl_per_job-1):
                    comb_i.append(nonseeds[i])
            comb.append(comb_i)
    return comb

def report_combination(outfile,seldata,combs):
    out = open(outfile,'w')

    nrun = {}
    tags = seldata.tags
    # report combination
    for comb in combs:
        seed = comb[0]
        iseed = seldata.poolids[seed]

        if iseed not in nrun:
            nrun[iseed] = 0
        nrun[iseed] += 1

        others = ''
        for tag in comb[1:]:
            others += ' %d'%(seldata.poolids[tag])
        out.write('SEED %d run %d: %s\n'%(iseed,nrun[iseed],others))

    # report bank
    for tag in tags:
        seedstr = ''
        if tag in seldata.seeds:
            seedstr = 'o'
        out.write('%-3d %3d %1s %s\n'%(seldata.poolids[tag],
                                       seldata.nuse[tag],seedstr,tag))
    out.close()

def launch_job(runinfo,combs,reconstruct=False,mode=''):
    out = open('alljobs','w')
    itstr = 'iter_%d'%runinfo.it

    ngen = runinfo.get_ngen_per_job(runinfo.phase)

    cmds = []
    for icomb,comb in enumerate(combs):
        iseed = runinfo.seldata.poolids[comb[0]]

        for k in range(ngen):
            script_vars_add = ''
            for i,strct in enumerate(comb):
                if i == 0 and k > 0 and reconstruct: # take partial of seed if k>0
                    script_vars_add += ' template%d=partial.%s.cons.pdb'%(i+1,strct)
                else:
                    script_vars_add += ' template%d=%s.pdb'%(i+1,strct)

            if len(comb) > 1: # cross
                cmd = runinfo.get_script('cross',runinfo.phase,itstr,iseed,script_vars_add,
                                         cstprefix=comb[0],
                                         autocst=(runinfo.opt.cross2_autocst and k%2==1))
            else: #mut
                cmd = runinfo.get_script('mut',runinfo.phase,itstr,iseed,script_vars_add,
                                         cstprefix=comb[0])
            cmds.append(cmd)
            
            out.write(cmd+'\n')
    out.close()

    n = 0
    t0 = time.time()
    while True:
        dt = (time.time()-t0)/60.0
        if dt > runinfo.opt.max_min_terminate:
            sys.exit('ERROR: jobs at iter_%d did not finish within %.1f minutes -- terminate!'%(runinfo.it,dt))
            
        if os.path.exists('gen.out'):
            n = len(utils.scparse('gen.out',['nuse']))-1
            if n >= runinfo.ngen(runinfo.phase): break
            
        os.system('echo "%s -j %d --workdir . :::: alljobs" > run.cmd'%(PARALLEL,runinfo.opt.nproc))
        os.system('%s -j %d --workdir . :::: alljobs > run.log 2> err'%(PARALLEL,runinfo.opt.nproc))
        time.sleep(5)

def run_iter(runinfo):
    ngen = runinfo.ngen(runinfo.phase)
    
    combs = make_combination(runinfo)
    report_combination('combination.log',runinfo.seldata,combs)

    if os.path.exists('gen.out'): #in case when terminated 
        ndone = len(utils.scparse('gen.out',[]))
        if ndone > runinfo.ngen(runinfo.phase):
            return
    elif os.path.exists('gen.retag.out'): #in case when terminated 
        ndone = len(utils.scparse('gen.retag.out',[]))
        if ndone > runinfo.ngen(runinfo.phase):
            return

    launch_job(runinfo,combs,reconstruct=runinfo.opt.reconstruct_every_iter)
    os.system('rm *log')

def run_iter_recomb(runinfo):
    print( "Running DL predictions for all..." )

    #direct call
    pdbs = glob.glob('iter*.?.pdb')+glob.glob('iter*.??.pdb')
    npzs = [pdb[:-4]+'.npz' for pdb in pdbs if os.path.exists('%s.npz'%pdb[:-4])]
    n = len(pdbs)
    if len(pdbs) != len(npzs):
        nproc = runinfo.opt.nproc
        os.system('python %s/ErrorPredictor.py -p %d ./ 1> DANmsa.log'%(DANPATH,nproc))
    
    all_idx = list(range(n))
    logout = open('combination.log','w')
    cross_from_DAN = recomb.find_partners.main(infolder='./',
                                               out=open('recomb1D.txt','w'),
                                               seeds=all_idx,
                                               logout=logout)
    logout.close()

    # make all-to-all recomb
    pdbs = ['iter%d.%d.pdb'%(runinfo.it,k) for k in range(n)]
    extraargs = ' -2D sub -relax dual' # opt.recomb is for recomb suggestion for hybrid; use 2D always

    if runinfo.opt.native != '':
        extraargs += ' -native %s'%runinfo.opt.native

    cmd = 'python %s/recomb/modeling.py %s %s -out gen.retag.out %s > recomb.log\n'

    jobs = open('alljobs','w')
    for seed in cross_from_DAN:
        for partner in cross_from_DAN[seed]:
            jobs.write(cmd%(MYPATH,seed+".pdb",partner+".pdb",extraargs))
    jobs.close()

    ngen = len(pdbs)*4
    t0 = time.time()
    while True:
        dt = (time.time()-t0)/60.0
        if dt > runinfo.opt.max_min_terminate:
            sys.exit('ERROR: jobs at iter_%d did not finish within %.1f minutes -- terminate!'%(runinfo.it,dt))
        if os.path.exists('gen.retag.out'):
            n = len(utils.scparse('gen.retag.out',['nuse']))-1
            if n >= ngen: break
            
        os.system('echo "%s -j %d --workdir . :::: alljobs" > run.cmd'%(PARALLEL,runinfo.opt.nproc))
        os.system('%s -j %d --workdir . :::: alljobs > run.log 2> err'%(PARALLEL,runinfo.opt.nproc))
        time.sleep(5)
            
def finalize(runinfo):
    # clear first
    os.system('rm -rf iter_*/pred iter_*/*cstgen.txt')

    final_iter = runinfo.opt.niter
    workpath = 'iter_%d'%(final_iter)
    os.mkdir(workpath)
    
    refpath = 'iter_%d'%(final_iter-1)
    if not os.path.exists('%s/sel.out'%refpath):
        sys.exit("Final output %s/sel.out does not exist!"%refpath)

    os.system('%s/source/bin/extract_pdbs%s'%(ROSETTAPATH,ROSETTASUFFIX)+\
              ' -in:file:silent ref.out 1> /dev/null 2>/dev/null')
    os.system('cp %s/sel.out %s/ref.out'%(refpath,workpath))
    os.chdir(workpath)

    tags = [l[:-4] for l in glob.glob('iter%d*pdb'%final_iter)]

    DAN_and_gen_restraints(tags, tags, outpath='pred', generate_cst=False, nproc=min(50,runinfo.opt.nproc))

def main(args):
    runinfo = RunInfo(args)
    utils.check_files(['iter_0/ref.out',
                       'input.fa','init.pdb','disulf.def','t000_.3mers','t000_.9mers',
                       '%s/source/bin/extract_pdbs%s'%(ROSETTAPATH,ROSETTASUFFIX),
                       '%s/source/bin/rosetta_scripts%s'%(ROSETTAPATH,ROSETTASUFFIX),
                       PARALLEL
                   ])

    for it in range(runinfo.opt.niter):
        runinfo.update_phase('iter_%d/PHASE'%it)
        if os.path.exists('iter_%d/DONE'%it): continue
        runinfo.it = it

        print("Preparing for iter %d..."%(it))
        gen_iter(runinfo)

        os.chdir('iter_%d'%it)

        print('%s/source/bin/extract_pdbs%s'%(ROSETTAPATH,ROSETTASUFFIX)+\
              ' -in:file:silent ref.out 1> /dev/null 2>/dev/null')
        os.system('%s/source/bin/extract_pdbs%s'%(ROSETTAPATH,ROSETTASUFFIX)+\
                  ' -in:file:silent ref.out 1> /dev/null 2>/dev/null')
        
        runinfo.update_dcut()

        if runinfo.is_recomb_iter(it):
            # make a separate logic
            #run_iter_recomb(runinfo)
            pass
        else:
            # DL portion comes here
            prepick(runinfo)
            run_iter(runinfo)

        postpick(runinfo)

        os.chdir(runinfo.curr)

    finalize(runinfo)
    os.chdir(runinfo.curr)

def setup(inputsilent,predf,workpath):
    args = []
    iQ = np.mean(np.load(predf)['lddt'])

    if '-cons' in sys.argv: #conservative mode
        args = ['-cstweight','1.0'
                '-e2cst_args',"-pcore 0.7 0.7 0.8"]
        
        simcut = min(60,max(30,iQ))
        dcut0 = 0.3*40.0/max(40.0,simcut)
        
    else: # default; i.e. aggressive mode
        simcut = min(60,max(30,iQ-30))
        dcut0 = 0.4*30.0/max(30.0,simcut)
        
    args += ['-dcut0',' %.3f'%dcut0]
    if os.path.exists('native.pdb'):
        args += ['-native','%s/native.pdb'%os.getcwd()]

    os.system('mkdir %s 2>/dev/null'%workpath)
    os.chdir(workpath)
    os.system('cp ../input.fa ../disulf.def ../init.pdb ./ 2>/dev/null')
    os.system('ln -s ../t000_.?mers ./ 2>/dev/null')
    os.system('mkdir iter_0 2>/dev/null')
    os.system('cp ../%s iter_0/ref.out'%(inputsilent))

    # Optional if exists any
    os.system('cp ../input.*cst ./ 2>/dev/null')
    
    os.chdir('..')

    return args
    
if __name__ == "__main__":
    inputsilent = sys.argv[1] #silent after initial div.
    predf = sys.argv[2] #DAN output for starting model
    workpath = sys.argv[3]
    args = setup(inputsilent,predf,workpath)

    os.chdir(workpath)
    main(args)
    

