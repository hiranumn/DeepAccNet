import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import inspect
import pickle
import sys

# Crossentropy calculation
def xentropy(pred, truth, epsilon=0.0001, axis=2):
    temp = np.multiply(truth, np.log(pred+epsilon))
    temp = np.sum(temp, axis=axis)
    temp = -1*temp
    return temp

def mse(pred, truth):
    return np.square(pred-truth)

def bxentropy(pred, truth, epsilon=0.0001, axis=2):
    pred = np.stack([pred, 1-pred], axis=-1)
    truth = np.stack([truth, 1-truth], axis=-1)
    return xentropy(pred, truth, axis=axis)

def starting_lddt_plot(raw_filename):
    
    # Load the file 
    with open(raw_filename, 'rb') as handle:
        data = pickle.load(handle)
    x = []
    lddt_loss = []
    esto_loss = []
    mask_loss = []
    count = 0 
    for p in data.keys():
        samples = [i for i in data[p].keys()]
        for s in samples:
            x.append(data[p][s]["starting_lddt"])
            esto_loss.append(np.mean(data[p][s]["esto_xent"]))
            lddt_loss.append(np.mean(data[p][s]["lddt_mse"]))
            mask_loss.append(np.mean(data[p][s]["mask_xent"]))

    x = np.array(x)
    lddt_loss = np.array(lddt_loss)
    esto_loss = np.array(esto_loss)
    mask_loss = np.array(mask_loss)

    lddt_loss_plot = {}
    esto_loss_plot = {}
    mask_loss_plot = {}

    # Digitization from 0.3 to 0.8 with 0.01 bins.
    digits = np.arange(0.3,0.80,0.01)
    digitized = np.digitize(x, digits)
    ticks = np.arange(0.3,0.81,0.01)

    temp = lddt_loss
    avg = np.array([np.mean(temp[digitized==i]) for i in range(max(digitized)+1)])
    std = np.array([np.std(temp[digitized==i]) for i in range(max(digitized)+1)])
    lddt_loss_plot = avg, std, ticks

    temp = esto_loss
    avg = np.array([np.mean(temp[digitized==i]) for i in range(max(digitized)+1)])
    std = np.array([np.std(temp[digitized==i]) for i in range(max(digitized)+1)])
    esto_loss_plot = avg, std, ticks

    temp = mask_loss
    avg = np.array([np.mean(temp[digitized==i]) for i in range(max(digitized)+1)])
    std = np.array([np.std(temp[digitized==i]) for i in range(max(digitized)+1)])
    mask_loss_plot = avg, std, ticks

    return {"lddt":lddt_loss_plot, "esto":esto_loss_plot, "mask":mask_loss_plot}

def prplot(models, proteins, datadir="/projects/casp/dldata/"):
    
    # Extracts predictions that are more than 
    # 0 sequence separation away and less than 35 with ditance. 
    def vectorize(y, preds, distance, sep=9, dist=35):
        digitizations = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]
        masses = [digitizations[0]]+[(digitizations[i]+digitizations[i+1])/2 for i in range(len(digitizations)-1)]+[digitizations[-1]]
        assert(y.shape==preds.shape)
        assert(y.shape[:2]==distance.shape)
        y_vec = []
        preds_vec = []
        for i in range(y.shape[0]):
            for j in range(i+1, y.shape[1]):
                # Sequence separation less than 9 starting distance less than 20 ang
                if abs(i-j) > sep and distance[i,j] < dist:
                    temp1 = y[i,j,:]
                    temp2 = preds[i,j,:]
                    y_vec.append(temp1[[4,5,6,7,8,9,10]])
                    preds_vec.append(temp2[[4,5,6,7,8,9,10]])
        return np.array(y_vec), np.array(preds_vec)

    # Transforms extracted vectors.
    def transform(m, step=0):
        if step==0:
            return m[:,3]
        elif step==1:
            return np.sum(m[:,[2,3,4]], axis=-1)
        elif step==2:
            return np.sum(m[:,[1,2,3,4,5]], axis=-1)
        elif step==3:
            return np.sum(m[:,[0,1,2,3,4,5,6]], axis=-1)
    
    ######################################
    # Extracting predictions that matter # 
    ######################################
    vector_y = []
    vector_preds = []
    for pname in proteins:
        # non-native files.
        files = [i for i in listdir(join(models[0], pname)) if i!="native.npz"][:10]
        for f in files:
            # Just making sure predictions are available for all models.
            if np.sum([isfile(join(models[k],pname,f)) for k in range(len(models))]) == len(models):
                temp = np.load(join(models[0], pname, f))
                temp = dict([(k, temp[k]) for k in temp.keys()])
                
                # Taking true and predicted estograms.
                truth = np.load(join(models[0],pname,f))["estogram_true"] 
                truth = truth.transpose([1,2,0])
                preds = np.mean([np.load(join(models[k],pname,f))["estogram"] for k in range(len(models))], axis=0)
                preds = preds.transpose([1,2,0])
                if isfile(join(datadir,pname,f[:-4]+".features.npz")):
                    distance = np.load(join(datadir,pname,f[:-4]+".features.npz"))["tbt"][0]
                else:
                    distance = np.load(join("/projects/casp/dldata_ref/",pname,f[:-4]+".features.npz"))["tbt"][0]
                y_vec, y_pred = vectorize(truth, preds, distance)
                vector_y.append(y_vec)
                vector_preds.append(y_pred)
    vector_y = np.concatenate(vector_y, axis=0)
    vector_preds = np.concatenate(vector_preds, axis=0)
    
    plotdata = {}
    # Do it for 4 different thresholding
    for i in range(4):
        y = transform(vector_y, step=i)
        preds = transform(vector_preds, step=i)
        r = []
        p = []
        for th in np.arange(0,1.05,0.05):
            temp = preds>=th
            # Dot products gets number of true-positives
            r.append(np.dot(y, temp)/np.sum(y))
            if np.sum(temp)!=0:
                p.append(np.dot(y, temp)/np.sum(temp))
            else:
                p.append(1)
        plotdata[i] = (r,p)
    return plotdata

# ESTOGRAMIFICATION
def get_estogram(XY, digitization=[-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]):
    (X,Y) = XY
    residual = X-Y
    estogram = np.eye(len(digitization)+1)[np.digitize(residual, digitization)]
    return estogram

def lddt(decoy, ref, cutoff=15, threshold=[0.5, 1, 2, 4]):
   
    # only use parts that are less than 15A in ref structure
    mask = ref < cutoff
    for i in range(mask.shape[0]):
        mask[i,i]=False
   
    # Get interactions that are conserved
    conservation = []
    for th in threshold:
        temp = np.multiply((np.abs(decoy-ref) < th), mask)
        conservation.append(np.sum(temp, axis=0)/np.sum(mask, axis=0))
       
    return np.mean(conservation, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Error predictor network prediction analyzer",
                                     epilog="v0.0.1")
    
    parser.add_argument("modelname",
                        action="store",
                        help="Modelname")
    
    parser.add_argument("--basefolder",
                        "-b",
                        action="store",
                        default="/projects/ml/DeepAccNet/",
                        help="Base folder that model is contained (Default: /projects/casp/estograms2/)")
    
    parser.add_argument("--outfolder",
                        "-o",
                        action="store",
                        default="/projects/ml/DeepAccNet/analysis/",
                        help="Distination of analysis (Default: /projects/casp/analysis2)")
    
    parser.add_argument("--ensemble",
                    "-ensemble",
                    action="store_true",
                    default=False,
                    help="Ensemble mode (Default: False)")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Run in a verbose mode (Default: False)")
    
    args = parser.parse_args()
    
    base = args.basefolder
    output_folder = args.outfolder
    modelname = args.modelname
    verbose = args.verbose
    
    if not args.ensemble:
        modelnames = [modelname]
    else:
        modelnames = [modelname+"_"+ i for i in ["best", "second", "third", "fourth"]]
    if verbose: print(modelnames)

    #dirs = listdir(join(base, modelnames[0]))
    dirs = np.load("/home/hiranumn/PyTorchDeepAccNet/data/analysis_list.npy")
    print(dirs)
    
    ignorelist = ["3ctrA"]
    
    dirs = [d for d in dirs if not d in ignorelist]
        
    if verbose: print("Datasets:", len(dirs), "proteins")
            
    ################################
    # Generating arrays for plots  #
    ################################
    pr_plots = prplot([join(base, m) for m in modelnames], dirs)
    
    plotdata = {"pr":pr_plots}
    
    outputname = join(output_folder, modelname+"_prdata.pkl")
    with open(outputname, 'wb') as handle:
        pickle.dump(plotdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return 0

if __name__== "__main__":
    main()