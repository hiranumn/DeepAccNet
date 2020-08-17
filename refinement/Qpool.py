import sys
import copy
import numpy as np
from pyrosetta import *
DLPATH = os.environ['DANPATH']
from QScorer import QScorer

class Struct:
    def __init__(self,ss): #silent struct as initialize
        self.ss = ss #store for full  info
        self.name = ss.decoy_tag()
        self.pose = Pose()
        ss.fill_pose(self.pose)
        self.Q = 0.0
        self.nuse = 0
        self.poolid = -1
        self.score = ss.get_energy("score") #Rosetta score

        nres = self.pose.size()
        xyzCA = np.zeros((nres,3))
        xyzCA_r = ss.get_CA_xyz()
        
        for i in range(nres):
            for k in range(3):
                xyzCA[i][k] = xyzCA_r(k+1,i+1) #fortran style
        d2map = np.array([[np.dot(xyzCA[i] - xyzCA[j],xyzCA[i]-xyzCA[j]) \
                           for i in range(nres)] for j in range(nres)])

        self.dmap = np.sqrt(d2map)
                
        if ss.has_energy("Q"): self.Q = ss.get_energy("Q")
        if ss.has_energy("nuse"): self.nuse = int(ss.get_energy("nuse"))
        if ss.has_energy("poolid"): self.poolid = int(ss.get_energy("poolid"))

    def get_score(self,key):
        if key == 'Q': return -self.Q
        elif key == 'score': return self.score
        return None
    
def arg_parser():
    import argparse
    params = argparse.ArgumentParser()

    params.add_argument("-refpdb", required=True, help="")
    params.add_argument("-silent_ref", required=True, help="")
    params.add_argument("-silent_gen", required=True, help="")
    params.add_argument("-silent_out", default="sel.out", help="")
    params.add_argument("-outprefix", default="sel", help="")

    params.add_argument("-simlimit", type=float, default=0.2, help="")
    params.add_argument("-dcut", type=float, default=0.3, help="")
    params.add_argument("-filterscore", default="score", help="")
    params.add_argument("-seeds", type=int, default=[],nargs='+',
                        help="")

    params.add_argument("-add_init_penalty", default=True,
                        action="store_true",help="")
    params.add_argument("-iha_cut",type=float,  default=0.3, 
                        help="")
    params.add_argument("-iha_penalty_slope", default=10.0, 
                        help="")
    params.add_argument("-preserve_nuse_sim", default=True,
                        action="store_true",help="")
    params.add_argument("-min_reset_unused", type=int, default=3,
                        help="")

    params.add_argument("-verbose", default=False,
                        action="store_true",help="")

    opt = params.parse_args()
    if opt.iha_cut > 1.0:
        sys.exit("-iha_cut should be within [0,1]!")
    return opt

def CAlddt(dmap1,dmap2):
    #use first one for contact list
    nres = len(dmap1)
    acc  = np.zeros(nres)
    norm = np.zeros(nres)+0.0001

    for i in range(nres-1):
        for j in range(i+1,nres):
            if dmap1[i][j] > 15: continue
            count = 0.0
            diff = abs(dmap1[i][j]-dmap2[i][j])
            for crit in [0.5,1.0,2.0,4.0]:
                if diff < crit: count += 0.25
            acc[i] += count
            acc[j] += count
            norm[i] += 1.0
            norm[j] += 1.0
        
    lddts = acc/norm
    return lddts,np.mean(lddts)

def pose2CAxyz(pose):
    # as numpy
    xyz = np.zeros((pose.size(),3))
    for ires in range(pose.size()):
        CAcoord = pose.residue(ires+1).xyz(2)
        for k in range(3):
            xyz[ires][k] = CAcoord[k]
    return xyz

def calc_init_dev_penalty( xyz, xyz_r, iha_cut, slope ):
    # calc "gdt-ha" scale to each other
    dxyz = xyz-xyz_r
    ddist = np.sqrt(np.sum(dxyz*dxyz,axis=1))
    (n05,n1,n2,n4) = (0,0,0,0)
    for d in ddist:
        if d <= 0.5: n05 +=1 
        if d <= 1: n1 +=1 
        if d <= 2: n2 +=1 
        if d <= 4: n4 +=1 
    iha_r = 0.25*(n05+n1+n2+n4)/len(dxyz) + 0.0001
    dha = iha_cut - iha_r
    penalty = 0.0
    f = (dha/iha_r - 0.03)
    if f > 0: penalty = min(1.0, slope*f*f)
    #no penalty if iha_r > iha_cut

    return 1.0-penalty, iha_r
        
def filter_similar( pool, simlimit, out, priority="score" ):
    filtered = []
    out.write("Filter similar within %d structurs using simlimit=%.2f & priority %s\n"%(len(pool),simlimit,
                                                                                  priority))
    for i,struct1 in enumerate(copy.copy(pool)):
        similar = []
        for j,struct2 in enumerate(filtered):
            sim = CAlddt( struct1.dmap, struct2.dmap )[1]
            if 1.0-sim < simlimit: similar.append((1.0-sim,j))
            if len(similar) > 5: break
            
        if similar == []:
            filtered.append(struct1)
            #out.write("append %s"%struct1.name)
        else:
            #Qs = [struct.get_score(priority) for struct in filtered]
            #iQmax = np.argmax(Qs)
            #Q2 = Qs[iQmax]
            similar.sort()
            dist,j = similar[0]
            Q2 = filtered[j].get_score(priority)
            if struct1.Q < Q2:
                filtered[j] = struct1
                #print("replace %s -> %s"%(struct2.name,struct1.name))
            else:
                #print("remove %s: dist to %s = %.3f"%(struct1.name,struct2.name,dist))
                pass
                
    out.write("Got %d -> %d after filter similar\n"%(len(pool),len(filtered)))
    return filtered #replace
    
def read_silent( silent_fn ):
    pools = []
    pis = rosetta.core.import_pose.pose_stream.SilentFilePoseInputStream(silent_fn)
    while pis.has_another_pose():
        pools.append( Struct(pis.next_struct()) )
    return pools

def write_silent( pool_out, silent_out, retag="sel" ):
    silentOptions = rosetta.core.io.silent.SilentFileOptions()
    silentOptions.in_fullatom(True)
    sfd = rosetta.core.io.silent.SilentFileData(silentOptions)

    for i,struct in enumerate(pool_out):
        ss = struct.ss
        ss.fill_pose(struct.pose)
        ss.decoy_tag("%s.%d"%(retag,i))
        ss.add_energy("nuse",struct.nuse) #replace
        ss.add_energy("poolid",struct.poolid) #replace
        ss.add_energy("Q",struct.Q) #replace
        #sfd.add_structure(ss)
        sfd.write_silent_struct(ss,silent_out) #add one-by-one
    #sfd.write_all(silent_out)

def CSA_selection( pool_in, pool_new, opt, out, verbose=False ):
    # sort pool_new from best energy

    Qs_new = [-struct.Q for struct in pool_new]
    pool_new_ordered_idx = np.argsort(Qs_new)

    #print([Qs_new[i] for i in pool_new_ordered_idx])
    #if opt.reverse_order: pool_ref.reverse()

    npool = len(pool_in)
    for i in pool_new_ordered_idx: 
        struct1 = pool_new[i]
        sim = np.array([CAlddt(struct1.dmap,struct2.dmap)[1] for struct2 in pool_in])

        #check -- take one with closest for speed? or scan all similar?
        # Scan all similar starting from the worst Q -- should be fast enough
        similar = [j for j in range(npool) if 1.0-sim[j] < opt.dcut]
        out.write("%-30s: Q=%.3f, found %2d similar"%(struct1.name,struct1.Q,len(similar)))

        replace_sim,replace_max,dist = (False,False,0.0)

        ## struct1 are new ones, struct2 are prv pool members
        if len(similar) == 0: #relace max 
            j = np.argmin([pool_in[j].Q for j in range(npool)])
            struct2 = pool_in[j]
            dist = 1.0-sim[j]
            if struct1.Q > struct2.Q:
                pool_in[j] = struct1
                replace_max = True
                pool_in[j].poolid = struct2.poolid

        else: #replace-similar
            ## CHANGE
            ## do not replace if any similar has better Q
            simQ = [pool_in[j].Q for j in range(npool) if j in similar]
            if max(simQ) > struct1.Q: 
                out.write("-> PassSim\n")
                continue

            # sort everytime to track pool update 
            # option A. from worst Q
            pool_ref_ordered_idx = np.argsort([pool_in[j].Q for j in range(npool)])
            # option B. from most similar
            pool_ref_ordered_idx = np.argsort(-sim)

            for j in pool_ref_ordered_idx: # from worst
                if j not in similar: continue
                struct2 = pool_in[j]
                dist = 1.0-sim[j]
                if struct1.Q > struct2.Q:
                    pool_in[j] = struct1 #nuse=0
                    if opt.preserve_nuse_sim: pool_in[j].nuse = struct2.nuse
                    replace_sim = True
                    pool_in[j].poolid = struct2.poolid
                    break

        if replace_max:
            out.write(" -> ReplaceMax: %40s -> Pool %2d; Q=%5.3f -> %5.3f (d-to-mostsimilar: %.3f)\n"%(struct1.name,struct2.poolid,
                                                                                                    struct2.Q,struct1.Q,
                                                                                                    dist))
        elif replace_sim:
            out.write(" -> ReplaceSim: %40s -> Pool %2d; Q %5.3f -> %5.3f\n"%(struct1.name,struct2.poolid,
                                                                            struct2.Q,struct1.Q))
        else:
            out.write("-> Pass\n")

    # sort by Q
    pool_out = []
    Qs = [-struct.Q for struct in pool_in]
    pool_out = [pool_in[i] for i in np.argsort(Qs)]
        
    # decide nuse
    reset = False
    unused_pool = [struct for struct in pool_out if struct.nuse == 0]
    if len(unused_pool) <= opt.min_reset_unused:
        for i,struct in enumerate( pool_out ):
            pool_out[i].nuse = 0
        reset = True
        out.write("RESET: Unused pool %2d -> 0\n"%len(unused_pool))
        
    return pool_out, reset

def update_library_seeds( refpose, pool_in, pool_new, opt, out ):
    init_xyz = pose2CAxyz(refpose)

    # 1. filter input first with [CB-lddt]
    pool_new = filter_similar( pool_new, simlimit=opt.simlimit, out=out,
                               priority=opt.filterscore )

    # 2. Allatom score
    poses_new = [struct.pose for struct in pool_new]
    out.write("Run DAN-msa on %d poses...\n"%len(poses_new))

    scorer = QScorer()
    scores = scorer.score( poses_new )
    out.write("DAN-msa done.\n")
    for i,score in enumerate(scores):
        pool_new[i].Q = scores[i]
        
    # 3. superimpose new to topscorer / add penalty
    for struct in pool_new:
        # this superposition isn't great...
        # gdtha error ranges > 10%
        rosetta.core.scoring.calpha_superimpose_pose( struct.pose, pool_in[0].pose )

        # optional, add deformation penalty 
        if opt.add_init_penalty:
            # call gdttm functionality inside Rosetta
            rosetta.protocols.mpi_refinement.add_poseinfo_to_ss(struct.ss,refpose,"_ref")
            ha_r = struct.ss.get_energy("gdtha_ref")*0.01 #100 -> 1 scale

            dha = opt.iha_cut - ha_r
            penalty = 0.0
            f = (dha/ha_r - 0.03)
            if f > 0: penalty = min(1.0, opt.iha_penalty_slope*f*f)
            mulf = 1.0-penalty

            out.write("penalty %s: HA-to-ref %.3f -> %.3f\n"%(struct.name,ha_r,mulf))
            struct.Q *= mulf

    # 4.Apply CSA logic
    pool_out,stat = CSA_selection( pool_in, pool_new, opt, out )
    return pool_out,stat

def main(opt,out=sys.stdout):
    out.write("Selection using dcut/simlimit/iha_cut: %.2f/%.2f/%.2f.\n"%(opt.dcut,opt.simlimit,opt.iha_cut))
    init('-mute all')
    refpose = pose_from_file(opt.refpdb)
    # read/store as silent
    pool_in = read_silent( opt.silent_ref )
    # add seed use info
    for i,struct in enumerate(pool_in):
        if struct.poolid in opt.seeds:
            struct.nuse += 1
        out.write("Ref struct %2d %-20s, nuse/poolid/Q: %2d %2d %5.3f\n"%(i,struct.name,
                                                                          struct.nuse,struct.poolid,struct.Q))
        
    pool_new = read_silent( opt.silent_gen )
    pool_out,reset = update_library_seeds( refpose, pool_in, pool_new, opt, out )

    write_silent( pool_out, opt.silent_out, retag=opt.outprefix )

    return reset
    
if __name__ == "__main__":
    opt = arg_parser()
    main(opt)
