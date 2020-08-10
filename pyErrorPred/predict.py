import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import time
from .resnet import *
from .model import *
from .deepLearningUtils import *
    
def getData(tmp, cutoff=0):
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
    _2d = np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
    _truth = None

    return _3d, _1d, _2d, _truth   

# VARIANCE REDUCTION
def transform(X, cutoff=4, scaling=3.0):
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
    
def predict(samples, modelpath, outfolder, num_blocks=5, num_filters=128, ensemble=False, verbose=False, csv=False):
    
    result = {}
    if csv:
        for s in samples:
            result[s] = []
    
    n_models = 5 if ensemble else 2
    for i in range(1, n_models):
        modelname = modelpath+"_rep"+str(i)
        if verbose: print("Loading", modelname)
        model = Model(obt_size = 70,
                      tbt_size = 33,
                      prot_size = None,
                      num_chunks = num_blocks,
                      channel = num_filters,
                      optimizer = "adam",
                      loss_weight = [1.0, 0.25, 10.0],
                      name = modelname,
                      label_smoothing = False,
                      no_last_dilation = True,
                      partial_instance_norm = True,
                      bert = False)
        model.load()
            
        for j in range(len(samples)):
            if verbose: print("Predicting for", samples[j], "(network rep"+str(i)+")") 
            tmp = join(outfolder, samples[j]+".features.npz")
            batch = getData(tmp)
            lddt, estogram, mask = model.predict(batch)
            
            if not csv:
                if not ensemble:
                    np.savez_compressed(join(outfolder, samples[j]+".npz"),
                                        lddt = lddt,
                                        estogram = estogram,
                                        mask = mask)
                else:
                    np.savez_compressed(join(outfolder, samples[j]+".rep"+str(i)+".npz"),
                                        lddt = lddt,
                                        estogram = estogram,
                                        mask = mask)
            else:
                result[samples[j]].append(np.mean(lddt))
                
    return result
                
def merge(samples, outfolder, verbose=False):
    for j in range(len(samples)):
        if verbose: print("Merging", samples[j])

        lddt = []
        estogram = []
        mask = []
        for i in range(1,5):
            temp = np.load(join(outfolder, samples[j]+".rep"+str(i)+".npz"))
            lddt.append(temp["lddt"])
            estogram.append(temp["estogram"])
            mask.append(temp["mask"])

        # Averaging
        lddt = np.mean(lddt, axis=0)
        estogram = np.mean(estogram, axis=0)
        mask = np.mean(mask, axis=0)

        # Saving
        np.savez_compressed(join(outfolder, samples[j]+".npz"),
                lddt = lddt,
                estogram = estogram,
                mask = mask)
                
def clean(samples, outfolder, noEnsemble=False, multimodel=False, verbose=False):
    if multimodel:
        os.remove(join(outfolder, "dist.npy"))
    for i in range(len(samples)):
        if verbose: print("Removing", join(outfolder, samples[i]+".features.npz"))
        os.remove(join(outfolder, samples[i]+".features.npz"))
        if not noEnsemble:
            for j in range(1,5):
                if verbose: print("Removing", join(outfolder, samples[i]+".rep"+str(j)+".npz"))
                os.remove(join(outfolder, samples[i]+".rep"+str(j)+".npz"))