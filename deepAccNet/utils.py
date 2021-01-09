import numpy as np
import os
from os import listdir
from os.path import join, isfile, isdir

# GET DATA
def getData(tmp, cutoff=0, bertpath=""):
    data = np.load(tmp)

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

    # 2D information
    orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
    orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
    euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
    maps = data["maps"]
    tbt = data["tbt"].T
    sep = seqsep(tbt.shape[0])

    # Transform input distance
    tbt[:,:,0] = transform(tbt[:,:,0])
    maps = transform(maps, cutoff=cutoff)

    _3d = (idx, val)
    _1d = (np.concatenate([angles, obt, prop], axis=-1), None)
    
    if bertpath!="":
        bert = np.load(bertpath)
        bert = np.transpose(bert, [1,2,0])
        _2d = np.concatenate([tbt, maps, euler, orientations, sep, bert], axis=-1)
    else:
        _2d = np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
    _truth = None

    return _3d, _1d, _2d, _truth

# GET DATA
def getData_from_dict(data, cutoff=0, bertpath=""):

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

    # 2D information
    orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
    orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
    euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
    maps = data["maps"]
    tbt = data["tbt"].T
    sep = seqsep(tbt.shape[0])

    # Transform input distance
    tbt[:,:,0] = transform(tbt[:,:,0])
    maps = transform(maps, cutoff=cutoff)

    _3d = (idx, val)
    _1d = (np.concatenate([angles, obt, prop], axis=-1), None)
    
    if bertpath!="":
        bert = np.load(bertpath)
        bert = np.transpose(bert, [1,2,0])
        _2d = np.concatenate([tbt, maps, euler, orientations, sep, bert], axis=-1)
    else:
        _2d = np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
    _truth = None

    return _3d, _1d, _2d, _truth

# VARIANCE REDUCTION
def transform(X, cutoff=4, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

# SEQUENCE SEPARATION
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def merge(samples, outfolder, per_res_only=False, verbose=False):
    for j in range(len(samples)):
        try:
            if verbose: print("Merging", samples[j])

            lddt = []
            estogram = []
            mask = []
            for i in ["best", "second", "third", "fourth"]:
                temp = np.load(join(outfolder, samples[j]+"_"+i+".npz"))
                lddt.append(temp["lddt"])
                if per_res_only:
                    continue
                estogram.append(temp["estogram"])
                mask.append(temp["mask"])

            # Averaging
            lddt = np.mean(lddt, axis=0)
            if not per_res_only:
                estogram = np.mean(estogram, axis=0)
                mask = np.mean(mask, axis=0)

            # Saving
            if per_res_only:
                np.savez_compressed(join(outfolder, samples[j]+".npz"),
                        lddt = lddt.astype(np.float16))
            else:
                np.savez_compressed(join(outfolder, samples[j]+".npz"),
                        lddt = lddt.astype(np.float16),
                        estogram = estogram.astype(np.float16),
                        mask = mask.astype(np.float16))
        except:
            print("Failed to merge for", join(outfolder, samples[j]+".npz"))
        
def clean(samples, outfolder, ensemble=False, verbose=False):
    for i in range(len(samples)):
        try:
            if verbose: print("Removing", join(outfolder, samples[i]+".features.npz"))
            if isfile(join(outfolder, samples[i]+".features.npz")):
                os.remove(join(outfolder, samples[i]+".features.npz"))
            if isfile(join(outfolder, samples[i]+".fa")):
                os.remove(join(outfolder, samples[i]+".fa"))
            if isfile(join(outfolder, "bert_"+samples[i]+".npy")):
                os.remove(join(outfolder, "bert_"+samples[i]+".npy"))
            if ensemble:
                for j in ["best", "second", "third", "fourth"]:
                    if verbose: print("Removing", join(outfolder, samples[i]+"_"+j+".npz"))
                    if isfile(join(outfolder, samples[i]+"_"+j+".npz")):
                        os.remove(join(outfolder, samples[i]+"_"+j+".npz"))
        except:
            print("Failed to clean for", samples[i])
