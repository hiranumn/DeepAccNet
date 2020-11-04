import sys,os
from pyrosetta import *
import numpy as np

MYPATH = os.path.dirname(os.path.abspath(__file__))
SCRIPTPATH = os.environ['SCRIPTPATH']
sys.path.insert(0,SCRIPTPATH)

import estogram2cst

def arg_parser(argv):
    import argparse
    
    parser = argparse.ArgumentParser\
             (description='')
    ## Input
    parser.add_argument('pdb1', help="reference pdb")
    parser.add_argument('pdb2', help="partner pdb")
    parser.add_argument('-s', default="global",
                        help="superposition option [local/global]")
    parser.add_argument('-2D', dest="twoD", default="none", 
                        help="utilize 2D estogram as restraints; [none/sub/both]")
    
    parser.add_argument('-prefix', default="", help="prefix of outputs")
    parser.add_argument('-cst', default='bounded',
                        help="how to derive cst [none/greedy]")
    parser.add_argument('-cstw', default=1.0, type=float,
                        help="restraint strength")
    parser.add_argument('-relax', default="dual", 
                        help="run relax afterwards [dual/cart]]; default=dual")
    parser.add_argument('-spline_Pcut', default=0.3, type=float,
                        help="Include spline only if max(esto) > this")
    
    parser.add_argument('-fmin', type=float, default=0.2,
                        help="min fraction stealing from pdb2")
    parser.add_argument('-fmax', type=float, default=0.5,
                        help="max fraction stealing from pdb2")
    parser.add_argument('-out', default=None,
                        help="output silent")
    parser.add_argument('-native', default=None, help="native structure")
    parser.add_argument('-score', default='ref2015_cart')

    if len(argv) < 1:
        parser.print_help()
        sys.exit(1)

    opt = parser.parse_args(argv)
    return opt

def rmsd_simple(pose1,pose2,reslist):
    rmsd = 0.0
    for res in reslist:
        xyz1 = pose1.residue(res).xyz(2)
        xyz2 = pose2.residue(res).xyz(2)
        dxyz = (xyz1-xyz2)
        rmsd += dxyz.length_squared()
    return np.sqrt(rmsd/len(reslist))

def get_common_base(Q1,Q2,f=0.8):
    k = -int(f*len(Q1))
    o1 = np.argsort(Q1)[k:]
    o2 = np.argsort(Q2)[k:]
    common = [i for i in o2 if i in o1]
    return common

def align_at_base(pose1,pose2,res_base):
    # superimpose around base
    base_amap = rosetta.core.id.AtomID_Map_AtomID() #rosetta.core.id.AtomID.BOGUS_ATOM_ID())
    rosetta.core.pose.initialize_atomid_map( base_amap, pose1, rosetta.core.id.AtomID.BOGUS_ATOM_ID() )

    for res in res_base:
        id1 = rosetta.core.id.AtomID(2,res) #CA
        base_amap.set( id1, id1 )
    rmsd = 0.0 #dummy... has to recalculate
    rosetta.core.scoring.superimpose_pose(pose2,pose1,base_amap,rmsd,False)
    #pose2.dump_pdb("super.pdb")

    # explicitly calculate rmsd
    return rmsd_simple(pose1,pose2,res_base)

def steal_coord_at_reg(pose1,pose2,reg,rmsdcut=1.0):
    #first measure similarity at region
    rmsd = rmsd_simple(pose1,pose2,reg)
    if rmsd < rmsdcut:
        print("skip %d-%d as of rmsd(%.1f) < %.1f"%(reg[0],reg[-1],rmsd,rmsdcut))
        return False

    for res in reg:
        for atmno in range(1,pose1.residue(res).natoms()+1):
            atomid = rosetta.core.id.AtomID(atmno,res)
            xyz2 = pose2.xyz(atomid)
            pose1.set_xyz(atomid,xyz2)

    #pose1.dump_pdb("switched%d-%d.pdb"%(reg[0],reg[-1]))
    return True

def get_dmap(pose):
    n = pose.size()
    dmap = np.zeros((n,n))
    for res1 in range(1,n+1):
        atm1 = "CB"
        if not pose.residue(res1).has("CB"): atm1 = "CA"
        for res2 in range(res1+1,n+1):
            atm2 = "CB"
            if not pose.residue(res2).has("CB"): atm2 = "CA"
            d = pose.residue(res1).xyz(atm1).distance(pose.residue(res2).xyz(atm2))
            dmap[res1-1,res2-1] = dmap[res2-1,res1-1] = d
    return dmap

# main function grabbing relevant restraints from estogram
def get_pair_restraints(pose,restraints,esto,dmap,
                        paircst="sub",
                        incl=[],excl=[],
                        Pcut=0.7,dcut=15.0):

    n = len(esto)
    if incl == [] and excl != []:
        incl = list(range(n)) # allow if at least i or j 
    
    for i in range(n):
        for j in range(n):
            if j-i < 3: continue
            
            (atm1,atm2) = (5,5) #CB
            (atm_i,atm_j) = ('CB','CB')
            if not pose.residue(i+1).has("CB"):
                atm1 = 2
                atm_i = 'CA'
            if not pose.residue(j+1).has("CB"):
                atm2 = 2
                atm_j = 'CA'
                
            id1 = rosetta.core.id.AtomID(atm1,i+1)
            id2 = rosetta.core.id.AtomID(atm2,j+1)
            
            if paircst == "both": #allow double counting at i,j!
                if i+1 in excl and j+1 in excl: continue #"and"
            else: 
                if i+1 in excl or j+1 in excl: continue #"or"
                   
            if not (i+1 in incl or j+1 in incl): continue #"nor"

            key = '%d-%d'%(i+1,j+1)

            P = np.sum(esto[i][j][6:9])
            d = dmap[i][j]
            
            if P < Pcut or d > dcut: continue
            pair = estogram2cst.Pair(esto[i][j],id1,id2,d,atm_i,atm_j)
            restraints.append(pair)

def apply_cst(pose,csttype,res_altered,
              cstfunc,
              esto1=None,esto2=None,
              dmap1=None,dmap2=None,
              sig=1.0,tol=1.0,
              paircst="sub",
              spline_Pcut=0.3,verb=False):

    import pyrosetta.rosetta.core.scoring.func as Func
    dmap_pose = get_dmap(pose)

    if csttype == "coord":
        for res in res_altered:
            atomid = rosetta.core.id.AtomID(2,res) #CA
            xyz = pose.xyz(atomid)
            
            if cstfunc == 'bounded':
                func = pyrosetta.rosetta.core.scoring.constraints.BoundFunc(0.0,1.0,sig,"")
            elif cstfunc == 'flath':
                func = Func.FlatHarmonicFunc(0.0,sig,tol)
            elif cstfunc == 'sigmoid':
                func = Func.SigmoidFunc(tol,5.0/sig)
            else:
                continue
            cst = rosetta.core.scoring.constraints.CoordinateConstraint( atomid, atomid, xyz, func )
            pose.add_constraint( cst )
    
    elif csttype == 'pair':
        restraints = []

        ## Do not filter by Pcut, filter by spline_Pcut below instead
        # first get restraints from seed; forget at altered res
        get_pair_restraints(pose, restraints,esto1,dmap1,paircst=paircst,excl=res_altered,Pcut=0.0,dcut=20.0)
        print("Possible restraints from seed: ", len(restraints))
        # append from partner
        get_pair_restraints(pose, restraints,esto2,dmap2,paircst=paircst,incl=res_altered,Pcut=0.0,dcut=20.0)
        print("Possible restraints from all: ", len(restraints))

        ncst = 0
        for pair in restraints:
            func = pair.estogram2spline(maxP_spline_on=spline_Pcut)
            if not func: continue # when max(estogram) < spline_Pcut
            
            cst = rosetta.core.scoring.constraints.AtomPairConstraint( pair.ids[0], pair.ids[1], func )
            pose.add_constraint( cst )
            ncst += 1

    print("Actual csts applied after filtering: %d"%(ncst))

def relax(pose,res_cst,cstw,score,relaxscript):
    # cartmin
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    sf_fa = create_score_function('ref2015_cart')
    if score == 'ref2015_hackmemb_cart':
        sf_fa.set_weight(rosetta.core.scoring.fa_rep, 0.8)
        sf_fa.set_weight(rosetta.core.scoring.fa_sol, 0.0)
        sf_fa.set_weight(rosetta.core.scoring.lk_ball_wtd, 0.0)
        sf_fa.set_weight(rosetta.core.scoring.fa_elec, 2.0)
        sf_fa.set_weight(rosetta.core.scoring.hbond_sr_bb, 2.0)
        sf_fa.set_weight(rosetta.core.scoring.hbond_lr_bb, 2.0)
        sf_fa.set_weight(rosetta.core.scoring.hbond_bb_sc, 2.0)
        sf_fa.set_weight(rosetta.core.scoring.hbond_sc, 2.0)

    if pose.constraint_set().has_constraints():
        sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, cstw)
        sf_fa.set_weight(rosetta.core.scoring.coordinate_constraint, cstw)
        
    relax = rosetta.protocols.relax.FastRelax(sf_fa,relaxscript)
    relax.set_movemap(mmap)
    relax.apply(pose)

# unused currently
def adjust_conn(regs,pose):
    dssp = rosetta.core.scoring.dssp.Dssp(pose)
    SSs = [a for a in dssp.get_dssp_reduced_IG_as_L_secstruct()] # as a string
    print("SS:",''.join(SSs))
    coils = [i for i,SS in enumerate(SSs) if SS == 'C']

    for ir,reg in enumerate(regs):
        (i,e) = (reg[0],reg[-1])
        if reg[0] > 1 and reg[0] not in coils: # extend to nearest coil
            nearest = [j for j in coils if j < i]
            if nearest != [] and i-max(nearest):
                i = max(nearest)
        if reg[-1] < pose.size() and reg[-1] not in coils: # extend to nearest coil
            nearest = [j for j in coils if j > e]
            if nearest != [] and min(nearest)-e:
                e = max(nearest)
        regs[ir] = range(i,e)

def get_super(Q1,Q2,threshold=0.03):
    superres = (Q2-Q1>threshold) #dimension
    for k in range(1,len(superres)-1):
        if superres[k-1] and superres[k+1]: superres[k] = True
    regs = []
    for k in range(len(superres)):
        if not superres[k]: continue
        if regs == [] or k-regs[-1][-1] > 1:
            regs.append([k+1])
        else:
            regs[-1].append(k+1)

    regs = [reg for reg in regs if len(reg) > 3] # at least 4
    nres = sum([len(reg) for reg in regs])
    return regs,nres

def main(pose1,pose2,npz1,npz2,opt=None):
    Q1 = np.load(npz1)['lddt']
    Q2 = np.load(npz2)['lddt']
    esto1 = np.load(npz1)['estogram']
    esto2 = np.load(npz2)['estogram']
    n = len(Q1)

    prefix = opt.prefix+'%s_%s'%(npz1[:-4],npz2[:-4])

    # superimpose
    if opt.s == 'local': #not tested
        res_base = get_common_base(Q1,Q2)
    else: #global; default
        res_base = list(range(1,n+1))
    print("superimpose around ", res_base)
    align_at_base(pose1,pose2,res_base)

    # get list of residues pose2 superior to pose1
    threshold = 0.03
    (nmin,nmax) = (int(n*opt.fmin),int(n*opt.fmax))
    for k in range(20): #dynamically adjust to match fraction within min<->max
        regs_super2,nsuper = get_super(Q1,Q2,threshold)
        if nsuper < nmin:
            threshold -= 0.03
        elif nsuper > nmax:
            threshold += 0.03
        else:
            break

    print("Fraction %.3f (cut=%.2f), Q2 super at"%(float(nsuper/n),threshold), regs_super2)

    # attach region by region
    res_altered = []
    pose_work = pose1
    for reg in regs_super2:
        stat = steal_coord_at_reg(pose_work,pose2,reg)
        if stat: res_altered += reg

    ## ADD 2D restraints here
    # Only if 2D requested
    if opt.twoD != 'none':
        dmap1 = get_dmap(pose1)
        dmap2 = get_dmap(pose2)

        apply_cst(pose_work,"pair",
                  res_altered,opt.cst,
                  esto1,esto2,dmap1,dmap2,
                  paircst=opt.twoD,
                  spline_Pcut=opt.spline_Pcut,
                  verb=True)

    relaxscript = None
    if opt.relax == 'dual':
        relaxscript = SCRIPTPATH+'/rosetta_scripts/dual.script'
    elif opt.relax == 'cart':
        relaxscript = SCRIPTPATH+'/rosetta_scripts/cart2.script'
    else:
        print("unknown relax mode; pass relax")

    if relaxscript != None:
        relax(pose_work,res_altered,opt.cstw,opt.score,relaxscript)

    # evaluation against native
    gdtmm = -1.0
    if opt.native != None:
        refpose = pose_from_file(opt.native)
        natseq = rosetta.core.sequence.Sequence( refpose.sequence(),"native",1 ) 
        seq    = rosetta.core.sequence.Sequence( pose_work.sequence(),"model",1 ) 
        aln = rosetta.core.sequence.align_naive(seq,natseq)
        gdtmm = rosetta.protocols.hybridization.get_gdtmm(refpose,pose_work,aln)
        
    if opt.out != None:
        silentOptions = rosetta.core.io.silent.SilentFileOptions()
        silentOptions.in_fullatom(True)
        sfd = rosetta.core.io.silent.SilentFileData(silentOptions)
        ss = rosetta.core.io.silent.BinarySilentStruct(silentOptions,pose_work,prefix)
        if gdtmm > 0.0:
            ss.add_energy("GDTMM_final",gdtmm) # in hybridize format
        sfd.write_silent_struct(ss,opt.out)
    else:
        pose_work.dump_pdb("%s.final.pdb"%prefix)

if __name__ == "__main__":
    init('-mute all') #pyrosetta
    opt = arg_parser(sys.argv[1:])
    
    pose1 = pose_from_file(opt.pdb1)
    pose2 = pose_from_file(opt.pdb2)
    npz1 = opt.pdb1[:-4]+'.npz'
    npz2 = opt.pdb2[:-4]+'.npz'
    main(pose1,pose2,npz1,npz2,opt)
