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
def xentropy(pred, truth, epsilon=0.0001, axis=0):
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
    data = {}
    
    for d in dirs:
        if verbose: print("Working on", d)
        samples = listdir(join(base, modelnames[0], d))
        prot = {}
        for s in samples:
            temp = {}
            _id = s[:-4] # id without .npz

            # Do ensembling if necessary
            if not args.ensemble:
                temp_pred = np.load(join(base, modelname, d, s))
                prediction = {}
                for tag in ["lddt", "esto", "mask", "lddt_true", "esto_true", "mask_true"]:
                    prediction[tag] = temp_pred[tag]
            else:
                predictions = [np.load(join(base, n, d, s)) for n in modelnames]
                prediction = {}
                for tag in ["lddt", "esto", "mask", "lddt_true", "esto_true", "mask_true"]:
                    prediction[tag] = np.mean([_[tag] for _ in predictions], axis=0)

            # Do ensembling if necessary
            temp["lddt_mse"] = mse(prediction["lddt"], prediction["lddt_true"])
            temp["esto_xent"] = xentropy(prediction["esto"], prediction["esto_true"], axis=2)
            temp["mask_xent"] = bxentropy(prediction["mask"], prediction["mask_true"], axis=2)
            temp["starting_lddt"] = np.mean(prediction["lddt_true"])
            temp["lddt"] = prediction["lddt_true"]
            temp["size"] = int(len(prediction["lddt"]))
            prot[_id] = temp

        data[d] = prot

    # Saving raw data containing per sample per interaction statistics.
    outputname = join(output_folder, modelname+"_raw.pkl")
    with open(outputname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #################################################
    # Averaging statistics for different conditions #
    #################################################
    labels = ["esto_xent", "mask_xent", "lddt_mse"]
    conds = [lambda s: True,
             lambda s: s["size"]<=80,
             lambda s: s["size"]>80 and s["size"]<=120,
             lambda s: s["size"]>120, 
             lambda s: s["starting_lddt"]<=.7, 
             lambda s: s["starting_lddt"]>.7, ]
    cond_labels = ["All", "Small size", "Medium size", "Large size", "Low-mid lddt", "High lddt"]
    
    result = {}
    index = 0
    for cond in conds:
        output = dict([(l,[]) for l in labels])
        for k in data.keys():
            prot = data[k]
            for n in prot.keys():
                sample = prot[n]
                if cond(sample):
                    for l in labels:
                        output[l].append(np.mean(sample[l]))
        output = [(k, np.mean(output[k])) for k in output.keys()]
        result[cond_labels[index]] = output
        index += 1
        
    outputname = join(output_folder, modelname+"_analyzed.pkl")
    with open(outputname, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    ################################
    # Generating arrays for plots  #
    ################################
    starting_lddt_plots = starting_lddt_plot(join(output_folder, modelname+"_raw.pkl"))
    plotdata = {"starting_lddt":starting_lddt_plots}
    
    outputname = join(output_folder, modelname+"_plotdata.pkl")
    with open(outputname, 'wb') as handle:
        pickle.dump(plotdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return 0

if __name__== "__main__":
    main()