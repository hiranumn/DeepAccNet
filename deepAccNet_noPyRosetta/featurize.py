# Import necessary libraries
import numpy as np
import math
import scipy
import scipy.spatial
import time
import sys
from scipy.spatial import distance, distance_matrix
#sys.path.insert(0, ".")
#import deepAccNet as dan

from .dataProcessingUtils import atypes
from .conversion import dict_3LAA_to_tip

def parse_pdbfile(pdbfile):
    file = open(pdbfile, "r")
    lines = file.readlines()
    file.close()
    
    lines = [l for l in lines if l.startswith("ATOM")]
    output = {}
    for line in lines:
        if line[13] != "H": 
            aidx = int(line[6:11])
            aname = line[12:16].strip()
            rname = line[17:20].strip()
            cname = line[21].strip()
            rindex = int(line[22:26])
            xcoord = float(line[30:38])
            ycoord = float(line[38:46])
            zcoord = float(line[46:54])
            occupancy = float(line[54:60])

            temp = dict(aidx = aidx,
                        aname = aname,
                        rname = rname,
                        cname = cname,
                        rindex = rindex,
                        x = xcoord,
                        y = ycoord,
                        z = zcoord,
                        occupancy = occupancy)

            residue = output.get(rindex, {})
            residue[aname] = temp
            output[rindex] = residue
        
    output2 = []
    keys = [i for i in output.keys()]
    keys.sort()
    for k in keys:
        temp = output[k]
        temp["rindex"] = k
        temp["rname"] = temp["CA"]["rname"]
        output2.append(temp)
        
    return output2

def get_coords(pose):

    nres = len(pose)

    # three anchor atoms to build local reference frame
    N = np.stack([np.array([pose[i]["N"]["x"], pose[i]["N"]["y"], pose[i]["N"]["z"]]) for i in range(nres)])
    Ca = np.stack([np.array([pose[i]["CA"]["x"], pose[i]["CA"]["y"], pose[i]["CA"]["z"]]) for i in range(nres)])
    C = np.stack([np.array([pose[i]["C"]["x"], pose[i]["C"]["y"], pose[i]["C"]["z"]]) for i in range(nres)])

    # recreate Cb given N,Ca,C
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466

    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = ca * a + cb * b + cc * c

    return N, Ca, C, Ca+Cb

def set_lframe(pdict):

    # local frame
    z = pdict['Cb'] - pdict['Ca']
    z /= np.linalg.norm(z, axis=-1)[:,None]

    x = np.cross(pdict['Ca']-pdict['N'], z)
    x /= np.linalg.norm(x, axis=-1)[:,None]

    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1)[:,None]

    xyz = np.stack([x,y,z])

    pdict['lfr'] = np.transpose(xyz, [1,0,2])
    
def get_dihedrals(a, b, c, d):

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

def get_angles(a, b, c):
    
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]
    
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]
    
    x = np.sum(v*w, axis=1)

    return np.arccos(x)

def set_neighbors6D(pdict):

    N = pdict['N']
    Ca = pdict['Ca']
    Cb = pdict['Cb']
    nres = pdict['nres']
    
    dmax = 20.0
    
    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    
    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    
    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
    
    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    
    pdict['dist6d'] = dist6d
    pdict['omega6d'] = omega6d
    pdict['theta6d'] = theta6d
    pdict['phi6d'] = phi6d
    
def set_neighbors3D(pdict):
    
    # get coordinates of all non-hydrogen atoms
    # and their types
    xyz = []
    types = []
    pose = pdict['pose']
    nres = pdict['nres']
    for i in range(nres):
        r = pose[i]
        rname = r["rname"]
        keys = [i for i in r.keys() if i != "rname" and i != "rindex"]
        for j in range(len(keys)):
            aname = r[keys[j]]["aname"]
            name = rname+'_'+aname
            if aname != 'NV' and aname != 'OXT' and name in atypes:
                xyz.append([r[keys[j]]["x"], r[keys[j]]["y"], r[keys[j]]["z"]])
                types.append(atypes[name])

    xyz = np.array(xyz)
    xyz_ca = pdict['Ca']
    lfr = pdict['lfr']

    # find neighbors and project onto
    # local reference frames
    dist = 14.0
    kd = scipy.spatial.cKDTree(xyz)
    kd_ca = scipy.spatial.cKDTree(xyz_ca)
    indices = kd_ca.query_ball_tree(kd, dist)
    idx = np.array([[i,j,types[j]] for i in range(len(indices)) for j in indices[i]])

    xyz_shift = xyz[idx.T[1]] - xyz_ca[idx.T[0]]
    xyz_new = np.sum(lfr[idx.T[0]] * xyz_shift[:,None,:], axis=-1)

    #
    # discretize
    #
    nbins = 24
    width = 19.2

    # total number of neighbors
    N = idx.shape[0]

    # bin size
    h = width / (nbins-1)
    
    # shift all contacts to the center of the box
    # and scale the coordinates by h
    xyz = (xyz_new + 0.5 * width) / h

    # residue indices
    i = idx[:,0].astype(dtype=np.int16).reshape((N,1))
    
    # atom types
    t = idx[:,2].astype(dtype=np.int16).reshape((N,1))
    
    # discretized x,y,z coordinates
    klm = np.floor(xyz).astype(dtype=np.int16)

    # atom coordinates in the cell it occupies
    d = xyz - np.floor(xyz)

    # trilinear interpolation
    klm0 = np.array(klm[:,0]).reshape((N,1))
    klm1 = np.array(klm[:,1]).reshape((N,1))
    klm2 = np.array(klm[:,2]).reshape((N,1))
    
    V000 = np.array(d[:,0] * d[:,1] * d[:,2]).reshape((N,1))
    V100 = np.array((1-d[:,0]) * d[:,1] * d[:,2]).reshape((N,1))
    V010 = np.array(d[:,0] * (1-d[:,1]) * d[:,2]).reshape((N,1))
    V110 = np.array((1-d[:,0]) * (1-d[:,1]) * d[:,2]).reshape((N,1))

    V001 = np.array(d[:,0] * d[:,1] * (1-d[:,2])).reshape((N,1))
    V101 = np.array((1-d[:,0]) * d[:,1] * (1-d[:,2])).reshape((N,1))
    V011 = np.array(d[:,0] * (1-d[:,1]) * (1-d[:,2])).reshape((N,1))
    V111 = np.array((1-d[:,0]) * (1-d[:,1]) * (1-d[:,2])).reshape((N,1))

    a000 = np.hstack([i, klm0, klm1, klm2, t, V111])
    a100 = np.hstack([i, klm0+1, klm1, klm2, t, V011])
    a010 = np.hstack([i, klm0, klm1+1, klm2, t, V101])
    a110 = np.hstack([i, klm0+1, klm1+1, klm2, t, V001])

    a001 = np.hstack([i, klm0, klm1, klm2+1, t, V110])
    a101 = np.hstack([i, klm0+1, klm1, klm2+1, t, V010])
    a011 = np.hstack([i, klm0, klm1+1, klm2+1, t, V100])
    a111 = np.hstack([i, klm0+1, klm1+1, klm2+1, t, V000])

    a = np.vstack([a000, a100, a010, a110, a001, a101, a011, a111])
    
    # make sure projected contacts fit into the box
    b = a[(np.min(a[:,1:4],axis=-1) >= 0) & (np.max(a[:,1:4],axis=-1) < nbins) & (a[:,5]>1e-5)]
    
    pdict['idx'] = b[:,:5].astype(np.uint16)
    pdict['val'] = b[:,5].astype(np.float16)
    
# In: pose, Out: distance maps with different atoms
def extract_multi_distance_map(pose):
    # Get CB to CB distance map use CA if CB does not exist
    x1 = get_distmaps(pose, atom1="CB", atom2="CB", default="CA")
    # Get CA to CA distance map
    x2 = get_distmaps(pose, atom1=dict_3LAA_to_tip, atom2=dict_3LAA_to_tip)
    # Get Tip to Tip distancemap
    x3 = get_distmaps(pose, atom1="CA", atom2=dict_3LAA_to_tip)
    # Get Tip to CA distancemap
    x4 = get_distmaps(pose, atom1=dict_3LAA_to_tip, atom2="CA")
    output = np.stack([x1,x2,x3,x4], axis=-1)
    return output

# Gets distance for various atoms given a pose
def get_distmaps(pose, atom1="CA", atom2="CA", default="CA"):
    psize = len(pose)
    xyz1 = np.zeros((psize, 3))
    xyz2 = np.zeros((psize, 3))
    for i in range(psize):
        r = pose[i]
        
        if type(atom1) == str:
            if atom1 in r:
                xyz1[i-1, :] = np.array([r[atom1]["x"], r[atom1]["y"], r[atom1]["z"]])
            else:
                xyz1[i-1, :] = np.array([r[default]["x"], r[default]["y"], r[default]["z"]])
        else:
            temp = atom1.get(r["rname"], default)
            xyz1[i-1, :] = np.array([r[temp]["x"], r[temp]["y"], r[temp]["z"]])
    
        if type(atom2) == str:
            if atom2 in r:
                xyz2[i-1, :] = np.array([r[atom2]["x"], r[atom2]["y"], r[atom2]["z"]])
            else:
                xyz2[i-1, :] = np.array([r[default]["x"], r[default]["y"], r[default]["z"]])
        else:
            temp = atom2.get(r["rname"], default)
            xyz2[i-1, :]  = np.array([r[temp]["x"], r[temp]["y"], r[temp]["z"]])

    return distance_matrix(xyz1, xyz2)
    
def init_pose(pose):
    pdict = {}
    pdict['pose'] = pose
    pdict['nres'] = len(pose)
    pdict['N'], pdict['Ca'], pdict['C'], pdict['Cb'] = get_coords(pose)
    set_lframe(pdict)
    set_neighbors6D(pdict)
    set_neighbors3D(pdict)
    return pdict

def process(args):
    filename, outfile, verbose = args
    #try:
    start_time = time.time()

    pose = parse_pdbfile(filename)

    pdict = init_pose(pose)

    maps = extract_multi_distance_map(pose)
    
    cbmap = np.expand_dims(get_distmaps(pose, atom1="CB", atom2="CB", default="CA"), 0)

    np.savez_compressed(outfile,
        idx = pdict['idx'],
        val = pdict['val'],
        tbt = cbmap,
        maps = maps.astype(np.float16))
    if verbose: print("Processed "+filename+" (%0.2f seconds)" % (time.time() - start_time))