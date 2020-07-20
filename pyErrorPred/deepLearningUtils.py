import os
import tensorflow as tf
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

class dataloader:
    
    def __init__(self,
                 proteins, # list of proteins to load
                 datadir = "/projects/casp/dldata/", # Base directory for all protein data
                 tapedir = "/projects/casp/tape_features/",
                 lengthmax = 500, # Limit to the length of proteins, if bigger we ignore.
                 dataset = [],
                 digits = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0],
                 verbose = False,
                 include_native = True,
                 distance_cutoff = 0,
                 bert = False,):
        
        self.n = {}
        self.samples_dict = {}
        self.sizes = {}
        
        self.digits = digits
        self.datadir = datadir
        self.tapedir = tapedir
        self.verbose = verbose
        self.include_native = include_native
        self.distance_cutoff = distance_cutoff
        self.bert = bert
            
        # Loading file availability
        temp = []
        for p in proteins:
            if not self.bert or isfile(join(self.tapedir, "tape_"+p+".npy")):
                path = datadir+p+"/"
                # Loading data by where it came from.
                samples_files = [f[:-13] for f in listdir(path) if isfile(join(path, f)) and "features.npz" in f]

                # Removing native if necessasry, not default behavior
                if not self.include_native:
                    samples_files = [s for s in samples_files if s != "native"]
                    
                # Randomize
                np.random.shuffle(samples_files)
                samples = samples_files
                
                # If more than one sample exists
                if len(samples) > 0:
                    length = np.load(path+samples[0]+".features.npz")["tbt"].shape[-1]
                    if length < lengthmax:
                        temp.append(p)
                        self.samples_dict[p] = samples
                        self.n[p] = len(samples)
                        self.sizes[p] = length
                
        # Make a list of proteins
        self.proteins = temp
        
        # Randomly ordered index
        self.index = np.arange(len(self.proteins))
        np.random.shuffle(self.index)
        self.cur_index = 0

    def next(self, transform=True, pindex=-1):
        
        # Choose one protein and one sample
        pname = self.proteins[self.index[self.cur_index]]
        if pindex == -1:
            pindex = np.random.choice(np.arange(self.n[pname]))
        sample = self.samples_dict[pname][pindex]
        psize = self.sizes[pname]
        data = np.load(join(self.datadir, pname, sample+".features.npz"))
        
        # 3D coordinate information
        idx = data["idx"]
        val = data["val"]
        
        # 1D information
        angles = np.stack([np.sin(data["phi"]),
                           np.cos(data["phi"]),
                           np.sin(data["psi"]),
                           np.cos(data["psi"])], axis=-1)
        obt = data["obt"].T
        prop = data["prop"].T
        
        # Get TAPE Features:
        bert = None
        if self.bert:
            bert = np.load(join(self.tapedir, "tape_"+pname+".npy"))
        
        # 2D information
        orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
        orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
        euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
        maps = data["maps"]
        tbt = data["tbt"].T
        sep = seqsep(psize)
        
        # Get target
        native = np.load(join(self.datadir,pname,"native.features.npz"))["tbt"][0]
        estogram = get_estogram((tbt[:,:,0], native), self.digits)
        
        # Transform input distance
        if transform:
            tbt[:,:,0] = f(tbt[:,:,0])
            maps = f(maps, cutoff=self.distance_cutoff)
        
        self.cur_index += 1
        if self.cur_index == len(self.proteins):        
            self.cur_index = 0 
            np.random.shuffle(self.index)
        
        _3d = (idx, val)
        _1d = (np.concatenate([angles, obt, prop], axis=-1), bert)
        _2d = np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
        _truth = (None, estogram, native)
            
        return _3d, _1d, _2d, _truth

# VARIANCE REDUCTION
def f(X, cutoff=4, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

# ESTOGRAMIFICATION
def get_estogram(XY, digitization):
    (X,Y) = XY
    residual = X-Y
    estogram = np.eye(len(digitization)+1)[np.digitize(residual, digitization)]
    return estogram

# SEQUENCE SEPARATION
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

# LABEL SMOOTHING
def apply_label_smoothing(x, alpha=0.2, axis=-1):
    minind = 0
    maxind = x.shape[axis]-1
    
    # Index of true and semi-true labels
    index = np.argmax(x, axis=axis)
    lower = np.clip(index-1, minind, maxind)
    higher = np.clip(index+1, minind, maxind)
    
    # Location-aware label smoothing
    true = np.eye(maxind+1)[index]*(1-alpha)
    semi_lower = np.eye(maxind+1)[lower]*(alpha/2)
    semi_higher= np.eye(maxind+1)[higher]*(alpha/2)
    
    return true+semi_lower+semi_higher