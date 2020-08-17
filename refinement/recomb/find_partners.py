import numpy as np

import os
from os import listdir
from os.path import isdir, isfile, join
import math,sys

# requires scipy installation
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.spatial.distance import *
from scipy.cluster.hierarchy import *

SCRIPTPATH = os.environ['SCRIPTPATH']
sys.path.insert(0,SCRIPTPATH)
import utils

def seriation(Z,N,cur_index):
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

def get_clusters(accvals, image=False, f=0.25, method="average"):
    # Create linkage matrix
    size = N = accvals.shape[0]
    distance_between_decoys = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distance_between_decoys[i,j] = np.mean(np.sqrt(np.square((accvals[i])-(accvals[j]))))
            
    condensed = squareform(distance_between_decoys)
    
    z = sch.linkage(condensed, method=method)
    order = seriation(z, N, N + N-2)
    cs = sch.fcluster(z, f*condensed.max(), criterion="distance")
    return cs #which cluster each member belongs to 

def cb_lddt(nat, pdb):
    def d2(crd1,crd2):
        val = 0.0
        for k in range(3):
            val += (crd1[k]-crd2[k])*(crd1[k]-crd2[k])
        return val
    
    natcrds = utils.pdb2crd(nat,'CB')
    deccrds = utils.pdb2crd(pdb,'CB')

    reslist = natcrds.keys()
    contacts = []
    for res1 in reslist:
        for res2 in reslist:
            if res1 >= res2: continue
            dis2 = d2(natcrds[res1],natcrds[res2])
            if dis2 < 225.0:
                contacts.append((res1,res2,math.sqrt(dis2)))

    lddts = {}
    for res in reslist:
        lddts[res] = []

    for res1,res2,dnat in contacts:
        crd1 = deccrds[res1]
        crd2 = deccrds[res2]
        d = math.sqrt(d2(crd1,crd2))

        count = 0.0
        diff = abs(dnat-d)
        for crit in [0.5,1.0,2.0,4.0]:
            if diff < crit: count += 0.25
        lddts[res1].append(count)
        lddts[res2].append(count)

    inds = [i-1 for i in reslist]
    vals = [sum(lddts[res])/len(lddts[res]) for res in reslist]
    return inds, vals

def sliding_improvement(best, base, window=10):
    output = []
    for i in range(0, len(base)-5, 5):
        t1 = best[i:i+window]
        t2 = base[i:i+window]
        output.append(np.mean(t1-t2))
    return np.max(output)

# Given a folder full of predictions,
# 1. performs hirarchical clustering
# 2. computes centroids
# 3. computes compatibility among centroids

def cluster(infolder, testmode=False, slide=False,
            verbose=False,
            ntrial=20, nmin=10, nmax=25):
    files = [f[:-4] for f in listdir(infolder) if f.endswith('.npz')]
    order = [int(f.split('.')[1]) for f in files]
    files = [files[i] for i in np.argsort(order)] #reorder by index

    dec_acc = []
    for f in files:
        filename = join(infolder, f+".npz")
        dec_acc.append(np.load(filename)["lddt"])
    dec_acc = np.array(dec_acc)

    f = 0.25
    for i in range(ntrial):
        assignment = get_clusters(dec_acc, f=f)
        c_num = np.max(assignment)
        if c_num < nmin: f -= 0.01
        elif c_num > nmax: f += 0.01
        else: break

    if verbose:
        print("threshold:", f)
        print("# samples:", len(files))
        print("# clusters:", c_num)

    # Calculate centroids by taking avaerage
    centroids = {}
    for i in range(1, c_num+1):
        centroids[i] = np.mean(dec_acc[assignment==i], axis=0)

    compat_matrix = np.zeros((c_num, c_num))
    for i in range(c_num):
        for j in range(c_num):
            # Take best possible recombination at each position
            temp_best = np.max(np.stack([centroids[i+1], centroids[j+1]]), axis=0)
            # Quantify improvement as mean lddt improvement
            if slide:
                improvement = sliding_improvement(temp_best, centroids[i+1])
            else:
                improvement = np.mean(temp_best-centroids[i+1])
            assert(improvement>=0)
            compat_matrix[i,j] = improvement
    
    return np.array(files), assignment, compat_matrix, dec_acc

def get_region_complementary(i,js,d,logout=sys.stdout):
    for j in js:
        super = (d[j]-d[i]>0.03)[0]
        for k in range(1,len(super)-1):
            if super[k-1] and super[k+1]: super[k] = True
            
        regs = []
        for k in range(len(super)):
            if not super[k]: continue
            if regs == [] or k-regs[-1][-1] > 1:
                regs.append([k])
            else:
                regs[-1].append(k)

# Given the output of above functio and a sample name of interest
# Chooses 4 (optinal number) samples that likely amend the weekness of the sample of interest.
def choose_mates(name, names, assignment, compat_matrix,
                 counts, maxcounts=99,
                 num=4, image=False, infolder="",
                 logout=sys.stdout):
    
    # Get index of sample of interest
    index = np.arange(len(names))[names==name]
    assert(len(index) == 1)
    index = index[0]
    
    # Get cluster of sample of interest
    cluster = assignment[index]
    
    # Get compatibility vector and get ordering of clusters from most compatible to least compatible
    compat_vector = compat_matrix[cluster-1, :]
    temp = [(compat_vector[i], i+1) for i in range(len(compat_vector))]
    temp.sort(reverse=True)
    compatible_clusters = []
    npick = min(np.max(assignment),num)
    while len(compatible_clusters) < npick:
        compatible_clusters = [c for i,c in temp if counts[c-1] < maxcounts] 
        maxcounts += 1 #relieve criteria if fails
    compatible_clusters = compatible_clusters[:npick]
        
    # Choose samples based on clusters
    output = []
    #logout.write("%s: compatible clusters (self %d),"%(name,cluster)+" ".join(compatible_clusters)+"\n")
    for c in compatible_clusters:
        n = np.random.choice(names[assignment==c])
        counts[c-1] += 1
        output.append(n)
    
    return output

def main(infolder,out=None,verbose=False,seeds=[],logout=sys.stdout):
    if verbose:
        print("Reading", infolder)
        print("Writing to", outfile)
    
    output = "#base, pair1, pair2, pair3, pair4\n"
    n, a, c, d = cluster(infolder, testmode=False, slide=True, verbose=verbose)

    if seeds != []: n_seed = [n[i] for i in seeds]
    else: n_seed = n
    counts = np.zeros(len(c))
    nmax_choose = int(0.4*len(n_seed)) # not more than 40%

    combs = {}
    for s in n_seed: # filename
        partners = choose_mates(s, n, a, c,
                                counts, maxcounts=nmax_choose,
                                image=False, infolder=infolder,
                                logout=logout)
        i_s = np.arange(len(n))[n==s]
        i_ps = [np.arange(len(n))[n==p] for p in partners]
        get_region_complementary(i_s,i_ps,d,logout=logout)
        
        output += ",".join([s]+partners)+"\n"
        combs[s] = partners

    if out != None:
        out.write(output)
    return combs #indices are modelno (e.g. 0 for iter0.0, 1 for iter0.1,...)
        
if __name__ == "__main__":
    infolder = sys.argv[1]
    outfile = sys.argv[2]
    out = open(outfile,'w')
    seeds = []
    if '-seed' in sys.argv:
        seeds = [int(word) for word in sys.argv[sys.argv.index('-seed')+1].split(',')]
    main(infolder,out,True,seeds)
    out.close()
