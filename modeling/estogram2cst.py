import sys,os,copy
import numpy as np
from math import log,sqrt
import pyrosetta as PR
import utils

CURR = os.getcwd()

MEFF=0.0001
DELTA = [-99.0,-17.5,-12.5,-7.0,-3.0,-1.5,-0.75,
         0.0,
         0.75,1.5,3.0,7.0,12.5,17.5,99.0]
MYPATH = os.path.dirname(os.path.realpath(__file__))
REFSTAT_FILE = MYPATH + '/refstat.dbin.seqsep.Q.txt'

##############################################
MAXD0_SPLINE = 35.0 # Max input model's (i,j) distance to generate spline restraints 
MIND = 4.0  # Minimum distance in spline x
DCAP = 40.0 # Maximum distance in spline x

FUNC = ['bounded','sigmoid'] #cen/fa for Pcen > PCORE[1]

PCON_BOUNDED = 0.9
MAXD0_FHARM = 20.0  # Max input model's (i,j) distance to generate flat-harmonic
TOL = [3.0,4.0,2.0,1.0] #Tolerance of flat-harmonic function
PCORE=[0.3,0.6,0.7,0.8] # at what P(abs(deltaD)<1) to apply flat-bottom harmonic? (will skip flat-bottom harmonic if lower then this value)
P_spline_on = 1.0 # never use
Sigma=[2.5,2.5,2.0,1.0] #Sigma of flat-harmonic function
kCEN = int((len(DELTA)-1)/2)

###########################################
# interface through pyrosetta

class Pair:
    def __init__(self,estogram,id1,id2,d0,atm_i,atm_j):
        self.ids = (id1,id2) #Rosetta AtomID
        self.atms = (atm_i,atm_j)
        self.estogram = estogram
        self.Pcorr = estogram #placeholder for uncorrected
        self.d0 = d0
        self.Pcen = sum(estogram[kCEN-1:kCEN+2])
        self.seqsep = abs(id1.rsd()-id2.rsd())
        self.is_ulr = False
        
    def estogram2spline(self,maxP_spline_on=0.0):
        maxP = max(self.Pcorr)
        if maxP <= maxP_spline_on: return False

        xs = self.d0+DELTA
        ys = -np.log(self.Pcorr+1.0e-4)
        ys = np.flip(ys,0) # make as ascending order in distance
        
        ys -= ys[7] #subtract center
        
        xs_v = PR.rosetta.utility.vector1_double()
        ys_v = PR.rosetta.utility.vector1_double()
        for k,x in enumerate(xs):
            if x < 1.0:
                xs[k] = 1.0
                ys[k] = 9.999
                continue
            xs_v.append(x)
            ys_v.append(ys[k])
        ys_v[1] = max(2.0,ys[2]+2.0) # soft repulsion
            
        func = PR.rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, 1.0, xs_v, ys_v )
        return func

###############################################
# Rosetta form restraints
def param2cst(tol,sig,d0,Pstr,func,extra='',w_relative=False):
    form_spl = 'SPLINE TAG %-30s 1.0 1.0 0.5 #%s\n' #W_SPLINE = 1.0
    form_h = 'FLAT_HARMONIC %5.1f %5.2f %5.2f #%s\n'
    form_b = 'BOUNDED %5.1f %5.1f 1.0 %5.2f #%s\n'
    form_sig = 'SUMFUNC 3 SCALARWEIGHTEDFUNC %6.3f SIGMOID %6.3f %6.3f SCALARWEIGHTEDFUNC %6.3f  SIGMOID %6.3f %6.3f CONSTANTFUNC %6.3f #%s\n' #(w,x1,m1,-w,x2,m2,2*w) == (5/t,t,s,-5/t,t,s,10/t)

    if func == 'bounded':
        funcstr = form_b%(d0-tol,d0+tol,sig,Pstr)
    elif func == 'fharm':
        funcstr = form_h%(d0-tol,d0+tol,sig,Pstr)
    elif func == 'sigmoid':
        w = 1.0
        if w_relative: w = 2.0/sig
        m = 5.0/sig
        x1 = d0-tol
        x2 = d0+tol
        funcstr = form_sig%(-w,x1,m,w,x2,m,w,Pstr) #(w,x1,m1,-w,x2,m2,shift)
    elif func == 'spline':
        funcstr = form_spl%(extra,Pstr)
    
    return funcstr

######################################
# ULR
def ulr2trim(ulr,nres):
    totrim = []
    ulr_by_reg = utils.list2part(ulr)
    
    for reg in ulr_by_reg:
        res1 = min(reg)
        res2 = max(reg)
        #if abs(res2-res1) > 5:
        #    if res2 < nres: res2 -= 2
        #    if res1 > 1: res1 += 2
        if abs(res2-res1) > 3:
            if res2 < nres: res2 -= 1
            if res1 > 1: res1 += 1
        totrim += range(res1,res2+1)
    return totrim

def estimate_lddtG_from_lr(pred):
    nres = len(pred)
    P0mean = np.zeros(nres)
    for i in range(nres):
        n = 0
        for j in range(nres):
            if abs(i-j) < 13: continue ## up to 3 H turns
            n += 1
            P0mean[i] += sum(pred[i][j][6:9]) #+-1
        P0mean[i] /= n
    return np.mean(P0mean)
        
# MAIN modification as using lddt pred as input now
def ULR_from_pred(pred,lddtG,
                  fmin=0.15,fmax=0.25,dynamic=False,mode='mean',verbose=False):
    nres = len(pred) #pred is lddt-per-res
    if dynamic: #make it aggressive!
        fmax = 0.3+0.2*(0.55-lddtG)/0.3 #lddtG' range b/w 0.25~0.55
        if fmax > 0.5: fmax = 0.5
        if fmax < 0.3: fmax = 0.3
        fmin = fmax-0.1
        print( "dynamic ULR: lddtPred/fmin/fmax: %8.5f %6.4f %6.4f"%(lddtG, fmin, fmax))

    # non-local distance accuracy -- removes bias from helices
    P0mean = np.zeros(nres)
    for i in range(nres):
        n = 0
        for j in range(nres):
            if abs(i-j) < 13: continue ## up to 3 H turns
            n += 1
            P0mean[i] += sum(pred[i][j][6:9]) #+-1
        P0mean[i] /= n
        
    #soften by 9-window sliding
    P0mean_soft = np.zeros(nres)
    for i in range(nres):
        n = 0
        for k in range(-4,5):
            if i+k < 0 or i+k >= nres: continue
            n += 1
            P0mean_soft[i] += P0mean[i+k]
        P0mean_soft[i] /= n
        if(verbose):
            print( "%3d %8.4f %8.4f"%(i+1, P0mean[i], P0mean_soft[i]) )
        
    P0mean = P0mean_soft
        
    lddtCUT = 0.3 #initial
    for it in range(50):
        factor = 1.1
        if it > 10: factor = 1.05
        
        is_ULR = [False for ires in range(nres)]
        for i,lddtR in enumerate(P0mean):
            if lddtR < lddtCUT: is_ULR[i] = True
            
        utils.sandwich(is_ULR,super_val=True,infer_val=False)
        utils.sandwich(is_ULR,super_val=False,infer_val=True)
        f = is_ULR.count(True)*1.0/len(is_ULR)

        if f < fmin: lddtCUT *= factor
        elif f > fmax: lddtCUT /= factor
        else: break

    ULR = []
    for i,val in enumerate(is_ULR):
        if val: ULR.append(i+1)
    return utils.trim_lessthan_3(ULR,nres)
   
###################################################
# Spline
def read_Pref(txt):
    Pref = {}
    refD = 0
    for l in open(txt).readlines():
        words = l[:-1].split()
        ibin = int(words[0])
        seqsep = int(words[2])
        Qbin = int(words[3])
        if ibin not in Pref: Pref[ibin] = {}
        if seqsep not in Pref[ibin]: Pref[ibin][seqsep] = {}
        Ps = [float(word) for word in words[5:]]
        Pref[ibin][seqsep][Qbin] = Ps
        refD = 3
    return Pref,refD
    
def P2spline(outf,d0,Ps,Ps_uncorrected):
    xs = []
    ys = []
    Pref = Ps[kCEN] 
    Ps_count = []
    Ps_count_u = []

    for k,P in enumerate(Ps):
        d = d0 - DELTA[k]
        E = -log((P+MEFF)/Pref)
        xs.append(d)
        ys.append(E)

    xs.reverse()
    ys.reverse()

    kss = [0]
    ksf = [len(DELTA)-1]
    ks_count = []
    for k,d in enumerate(xs):
        if d-0.1 < MIND:
            if k not in kss:
                kss.append(k)
        elif d+5.0 > DCAP:
            if k not in ksf:
                ksf.append(k)
        else:
            ks_count.append(k)

    xs = [3.0,4.0] + [xs[k] for k in ks_count] + [40.0]
    Ps_count = [sum([Ps[-k-1] for k in kss])] + [Ps[-k-1] for k in ks_count] + [sum([Ps[-k-1] for k in ksf])]
    Ps_count_u = [sum([Ps_uncorrected[-k-1] for k in kss])] + [Ps_uncorrected[-k-1] for k in ks_count] + [sum([Ps_uncorrected[-k-1] for k in ksf])]
    y1 = -log((Ps_count[0]+MEFF)/Pref)
    ys = [max(3.0,y1+3.0),y1]+ [ys[k] for k in ks_count] + [0.0] #constant fade
        
    out = open(outf,'w')
    out.write('#d0: %8.3f\n'%d0)
    out.write('#P '+'\t%7.3f'*len(Ps)%tuple([d0-D for D in DELTA])+'\n')
    out.write('#P '+'\t%7.5f'*len(Ps)%tuple(Ps)+'\n')
    out.write('#Pu'+'\t%7.5f'*len(Ps_uncorrected)%tuple(Ps_uncorrected)+'\n')
    out.write('x_axis'+'\t%7.2f'*len(xs)%tuple(xs)+'\n')
    out.write('y_axis'+'\t%7.3f'*len(ys)%tuple(ys)+'\n')
    out.close()

# Main func for cst
def estogram2cst(dat,pdb,cencst,facst,
                 weakcst,
                 ulr=[],
                 do_reference_correction=False, # use it only for sm version,
                 Pcore=PCORE,func=FUNC,
                 Anneal=1.0,w_relative=False,
                 exulr_from_spline=False
):
    if (weakcst=="spline") and not os.path.exists('splines'):
        os.mkdir('splines')

    Q = np.mean(dat['lddt'])
    dat = dat['estogram']
    nres = len(dat)
    d0mtrx = utils.read_d0mtrx(pdb)
    aas = utils.pdb2res(pdb)
    Prefs = None
    
    if do_reference_correction:
        Prefs,refD = read_Pref(REFSTAT_FILE) 
        Qbin = min(5,max(0,int(Q-0.4)/0.1))

    nharm_cst_lr = 0
    soft_cst_info = []
    pdbprefix = pdb.split('/')[-1][:-4]
    
    for i in range(nres-4):
        res1 = i+1
        atm1 = 'CB'
        if aas[res1]== 'GLY': atm1 = 'CA'
        
        for j in range(nres):
            if j-i < 4: continue
            res2 = j+1
            #is_in_ulr = (res1 in ulr) or (res2 in ulr)
            is_in_ulr = False
            for k in range(-2,3):
                if res1+k in ulr:
                    is_in_ulr = True
                    break
                if res2+k in ulr:
                    is_in_ulr = True
                    break
            
            d0 = d0mtrx[i][j]
            
            seqsep = min(50,abs(i-j))
            if seqsep > 20:
                seqsep = int(20+(seqsep-20)/2) #trimmed version

            P1 = dat[i][j]
            P2 = dat[j][i]
            Pavrg = [0.5*(P1[k]+P2[k]) for k in range(len(P1))]

            if do_reference_correction:
                dbin = int(d0-4.0)
                if seqsep < 10:  continue
                if dbin <0: dbin = 0
                elif dbin >= max(Prefs.keys()): dbin = max(Prefs.keys())
                Pref = Prefs[dbin][seqsep][Qbin]
                Pcorrect = [P/(Pref[k]+0.001) for k,P in enumerate(Pavrg)]
            else:
                Pcorrect = Pavrg
                
            Pcorrect = [P/sum(Pcorrect) for P in Pcorrect] #renormalize
            
            if d0 > MAXD0_SPLINE: continue # 35.0 ang
            
            atm2 = 'CB'
            if aas[res2] == 'GLY': atm2 = 'CA'
            if seqsep < 4: continue

            Pcontact = 0.0
            for k,P in enumerate(Pcorrect):
                if d0 - DELTA[k] < 8.0:
                    Pcontact += P
                elif d0 - DELTA[k] < 10.0:
                    Pcontact += 0.5*P

            # sum of 3-contiguous P after correction
            maxP = max([sum(Pcorrect[k:k+2]) for k in range(1,len(DELTA)-1)])

            Pcen = np.sum(Pavrg[kCEN-1:kCEN+2]) #from uncorrected

            aa1 = utils.aa3toaa1(aas[res1])
            aa2 = utils.aa3toaa1(aas[res2])
            cstheader = 'AtomPair %3s %3d %3s %3d '%(atm1,res1,atm2,res2)
            
            #Spline
            # check
            if weakcst == 'spline' and not (is_in_ulr and exulr_from_spline):
                splf="./splines/%s.%d.%d.txt"%(pdbprefix,res1,res2)
                if not os.path.exists(splf): 
                    P2spline(splf,d0,Pcorrect,Pavrg) #always generate
                #spline for every pair < 35.0 Ang
                censtr = cstheader + param2cst(0,0,0,"Pmax %6.3f"%maxP,'spline',extra=splf)
                cencst.write(censtr)

                if weakcst == 'bounded' and Pcontact > PCON_BOUNDED and Pcen < Pcore[1]:
                    censtr = cstheader + param2cst(4.0,1.0,8.0,"Pcon %6.3f"%Pcontact,'bounded')
                    cencst.write(censtr)
                
            #core part: flat bottom harmonic
            if d0 > MAXD0_FHARM: continue

            if Pcen > Pcore[1]:
                if Pcen > Pcore[3]:
                    censtr = cstheader + param2cst(TOL[3],Sigma[3],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                    fastr  = cstheader + param2cst(TOL[3],Sigma[3],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                elif Pcen > Pcore[2]:
                    censtr = cstheader + param2cst(TOL[2],Sigma[2],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                    fastr  = cstheader + param2cst(TOL[2],Sigma[2],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                elif Pcen > Pcore[1]:
                    censtr = cstheader + param2cst(TOL[1],Sigma[1],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                    fastr  = cstheader + param2cst(TOL[1],Sigma[1],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                    
                cencst.write(censtr)
                facst.write(fastr)
                if seqsep >= 9:
                    nharm_cst_lr += 1
                    
            elif Pcen > Pcore[0] and seqsep >= 9:
                # store as list and not apply yet
                censtr = cstheader + param2cst(TOL[0],Sigma[0],d0,"Pcen %6.3f"%Pcen,func[0],w_relative=w_relative)
                fastr  = cstheader + param2cst(TOL[0],Sigma[0],d0,"Pcen %6.3f"%Pcen,func[1],w_relative=w_relative)
                soft_cst_info.append((Pcen,censtr,fastr))

    npair_cut = log(float(nres))*float(nres)
    if nharm_cst_lr < npair_cut:
        soft_cst_info.sort()
        nadd = 0
        ## NOTE: npair_cut*2 is very likely to destroy topology unless highly restrained
        #while nharm_cst_lr < npair_cut*2 and len(soft_cst_info) > 0:  ## too conservative -- reduce
        while nharm_cst_lr < npair_cut and len(soft_cst_info) > 0: 
            (P,censtr,fastr) = soft_cst_info.pop()
            cencst.write(censtr)
            facst.write(fastr)
            nharm_cst_lr += 1
            nadd += 1
        print( 'Not enough lr cst %d (cut %d): supplement with lower-Pcen %d csts'%(nharm_cst_lr-nadd, 
                                                                                    npair_cut,
                                                                                    nharm_cst_lr))
            
# main func for ulr & partial pdb
def dat2ulr(pdb,pred,lddtG):
    outprefix = pdb.split('/')[-1].replace('.pdb','')
    
    nres = len(pred)

    verbose = ('-verbose' in sys.argv)
    ulrs = [[] for k in range(3)]
    ulrs[0] = ULR_from_pred(pred,lddtG,fmin=0.10,fmax=0.20,verbose=verbose) 
    ulrs[1] = ULR_from_pred(pred,lddtG,fmin=0.40,fmax=0.50,verbose=verbose) #unused
    ulrs[2] = ULR_from_pred(pred,lddtG,dynamic=True,verbose=verbose)

    totrim = [[] for k in range(3)]
    for k,ulr in enumerate(ulrs):
        if ulr != []:
            totrim[k] = ulr2trim(ulr,nres)
    
    print( "ULR P.std : %s %d"%(pdb,len(ulrs[0])), ulrs[0] )
    if os.path.exists('partial.%s.cons.pdb'%outprefix):
        print("skip partial pdb generation as it exists: partial.%s.cons.pdb"%outprefix)
    else:
        utils.pdb_in_resrange(pdb,'partial.%s.cons.pdb'%outprefix,totrim[0],exres=True)

    print( "ULR P.aggr: %s %d"%(pdb,len(ulrs[2])), ulrs[2] ) #== dynamic mode
    if os.path.exists('partial.%s.aggr.pdb'%outprefix):
        print("skip partial pdb generation as it exists: partial.%s.aggr.pdb"%outprefix)
    else:
        utils.pdb_in_resrange(pdb,'partial.%s.aggr.pdb'%outprefix,totrim[2],exres=True)
    return ulrs

def main(npz,pdb,cstprefix=None):
    dat = np.load(npz)
    lddtG = np.mean(dat['lddt'])
    lddtG_lr = estimate_lddtG_from_lr(dat['estogram']) #this is long-range-only lddt
    ulrs = dat2ulr(pdb,dat['estogram'],lddtG_lr)
    
    ulr_exharm,weakcst,refcorr = ([],'spline',True)
    if '-exulr_from_harm' in sys.argv:
        ulr_exharm = ulrs[2]
    if '-weakcst' in sys.argv:
        weakcst = sys.argv[sys.argv.index('-weakcst')+1]
    if cstprefix == None:
        cstprefix = pdb.replace('.pdb','')
        
    w_relative = False
    if '-pcore' in sys.argv:
        Pcore_in = [0.4,
                    float(sys.argv[sys.argv.index('-pcore')+1]),
                    float(sys.argv[sys.argv.index('-pcore')+2]),
                    float(sys.argv[sys.argv.index('-pcore')+3])]
    else:
        Pcore_in = PCORE

    cencst = open(cstprefix+'.cst','w')
    facst = open(cstprefix+'.fa.cst','w')

    estogram2cst(dat,pdb,cencst,facst,
                 weakcst,#add_spline=(weakcst=='spline'),
                 ulr=ulr_exharm, #exclude fharm on dynamic-ulr
                 do_reference_correction=refcorr,
                 Pcore=Pcore_in,
                 func=FUNC,
                 Anneal=1.0)

    cencst.close()
    facst.close()
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print( "usage: python estogram2cst.py [npy file] [reference pdb file]" )
        sys.exit()
        
    npz = sys.argv[1]
    pdb = sys.argv[2]
    outprefix = sys.argv[3]
    main(npz,pdb,outprefix)
