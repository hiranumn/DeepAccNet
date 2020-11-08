import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join, isfile, isdir
from os import listdir

class DecoyDataset(Dataset):
    """Rosetta 800k decoy dataset."""

    def __init__(self,
                 targets,
                 root_dir        = "/projects/casp/dldata/",
                 multi_dir       = False,
                 root_dirs       = ["/projects/casp/dldata/", "/projects/casp/dldata_ref/"],
                 bert_dir        = "/home/hiranumn/for/justas/embeddings/",
                 lengthmax       = 500, 
                 digits          = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0],
                 verbose         = False,
                 include_native  = True,
                 distance_cutoff = 0,
                 bert            = False,
                 features        = []):
        
        # Properties
        self.datadir = root_dir
        self.digits = digits
        self.verbose = verbose
        self.include_native = include_native
        self.distance_cutoff = distance_cutoff
        self.lengthmax = lengthmax
        self.bert = bert
        self.bertdir = bert_dir
        self.multi_dir = multi_dir
        self.root_dirs = root_dirs
        self.features = features
        
        # stats
        self.n = {}
        self.samples_dict = {}
        self.sizes = {}
            
        # Loading file availability
        temp = []
        for p in targets:
            if not self.bert or isfile(join(self.bertdir, "bert_"+p+".npy")):
                # Loading data by where it came from.
                if not multi_dir:
                    path = join(self.datadir, p)
                    sample_files = [join(path, f[:-13]) for f in listdir(path) if isfile(join(path, f)) and "features.npz" in f]
                else:
                    sample_files = []
                    for directory in root_dirs:
                        path = join(directory, p)
                        if isdir(path):
                            sample_files += [join(path, f[:-13]) for f in listdir(path) if isfile(join(path, f)) and "features.npz" in f]
                
                # Removing native if necessasry. This is not a default behavior
                if not self.include_native:
                    sample_files = [s for s in sample_files if s.split("/")[-1] != "native"]

                # Randomize
                np.random.shuffle(sample_files)
                samples = sample_files

                # If more than one sample exists
                if len(samples) > 0:
                    length = np.load(samples[0]+".features.npz")["tbt"].shape[-1]
                    if length < self.lengthmax:
                        temp.append(p)
                        self.samples_dict[p] = samples
                        self.n[p] = len(samples)
                        self.sizes[p] = length
                
        # Make a list of proteins
        self.proteins = temp

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx, transform=True, pindex=-1):
        
        # Convert index
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get protein name 
        pname = self.proteins[idx]
        # Get sample name
        if pindex == -1:
            pindex = np.random.choice(np.arange(self.n[pname]))
        sample = self.samples_dict[pname][pindex]
        psize = self.sizes[pname]
        # Get data
        data = np.load(sample+".features.npz")
        
        # Get 3D coordinate information
        idx = data["idx"]
        val = data["val"]
        
        # Get 1D information
        angles = np.stack([np.sin(data["phi"]),
                           np.cos(data["phi"]),
                           np.sin(data["psi"]),
                           np.cos(data["psi"])], axis=-1)
        obt = data["obt"].T
        prop = data["prop"].T
        
        # Get 2D information
        orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
        orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
        euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
        maps = data["maps"]
        tbt = data["tbt"].T
        sep = self.seqsep(psize)
        
        # Get truth
        native = np.load(join(self.datadir,pname,"native.features.npz"))["tbt"][0]
        estogram, estogram_1hot = self.get_estogram((tbt[:,:,0], native), self.digits)
        
        # Get Transform input distance
        if transform:
            tbt[:,:,0] = self.dist_transform(tbt[:,:,0])
            maps = self.dist_transform(maps, cutoff=self.distance_cutoff)
        
        # Sort the features
        _1d = np.concatenate([angles, obt, prop], axis=-1)
        
        # Get Bert Features:
        if self.bert:
            bert = np.load(join(self.bertdir, "bert_"+pname+".npy"))
            bert = np.transpose(bert, [1,2,0])
            _2d = np.concatenate([tbt, maps, euler, orientations, sep, bert], axis=-1)
        else:
            _2d = np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
        _2d = np.expand_dims(_2d.transpose(2,0,1), 0)
        estogram = np.expand_dims(estogram, 0)
        estogram_1hot = np.expand_dims(estogram_1hot.transpose(2,0,1), 0)
        mask = native < 15
        
        if len(self.features) > 0:
            inds1d, inds2d = self.getMask(self.features)
            _1d = _1d[:, inds1d]
            _2d = _2d[:, inds2d, :, :]
        
        sample = {'idx': idx.astype(np.int32),
                  'val': val.astype(np.float32),
                  '1d': _1d.astype(np.float32), 
                  '2d': _2d.astype(np.float32), 
                  'estogram': estogram, 
                  'estogram_1hot': estogram_1hot,
                  'mask': np.expand_dims(mask.astype(np.float32), 0)}
            
        return sample
    

    # VARIANCE REDUCTION
    def dist_transform(self, X, cutoff=4, scaling=3.0):
        X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
        return np.arcsinh(X_prime)/scaling

    # ESTOGRAMIFICATION
    def get_estogram(self, XY, digitization):
        (X,Y) = XY
        residual = X-Y
        estogram = np.digitize(residual, digitization)
        estogram_1hot = np.eye(len(digitization)+1)[estogram]
        return estogram, estogram_1hot

    # SEQUENCE SEPARATION
    def seqsep(self, psize, normalizer=100, axis=-1):
        ret = np.ones((psize, psize))
        for i in range(psize):
            for j in range(psize):
                ret[i,j] = abs(i-j)*1.0/normalizer-1.0
        return np.expand_dims(ret, axis)
    
    # Getting masks
    def getMask(self, include):
        feature2D = [("distance", 1), ("rosetta", 9), ("distance2", 4), ("orientation", 18), ("seqsep", 1), ("bert", 16)]
        feature1D = [("angles", 10), ("rosetta", 4), ("ss", 4), ("aa", 52)]
        for e in include:
            if e not in [i[0] for i in feature2D] and e not in [i[0] for i in feature1D]:
                print("Feature names do not exist.")
                print([i[0] for i in feature1D])
                print([i[0] for i in feature2D])
                return -1
        mask = []
        temp = []
        index = 0
        for f in feature1D:
            for i in range(f[1]):
                if f[0] in include: temp.append(index)
                index+=1
        mask.append(temp)
        temp = []
        index = 0
        for f in feature2D:
            for i in range(f[1]):
                if f[0] in include: temp.append(index)
                index+=1
        mask.append(temp)
        return mask