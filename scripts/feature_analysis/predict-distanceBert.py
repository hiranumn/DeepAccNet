import sys
sys.path.insert(0, "/home/hiranumn/PyTorchDeepAccNet/")
import deepAccNet as dan

import torch
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, isfile, isdir, basename, normpath
from os import listdir
import os
import argparse

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Error predictor network",
                                     epilog="v0.0.1")
    parser.add_argument("name",
                        action="store",
                        help="name")
    
    parser.add_argument("--bert",
                        "-bert",
                        action="store_true",
                        default=False,
                        help="Running on a single pdb file instead of a folder (Default: False)")
    
    args = parser.parse_args()
    
    bert = args.bert
    name = args.name
    distination = "/projects/ml/DeepAccNet/"
    multi_dir = True
    lengthmax = 280
    
    ################################
    features = ["distance", "distance2", "bert"]
    ################################
    
    for model in ["best", "second", "third", "fourth", "fifth"]:
        # Load the best model
        
        net = dan.DeepAccNet_no3Dno1D(num_chunks   = 5,
                                        num_channel  = 128,
                                        onebody_size = 0,
                                        twobody_size = 21)
        
        checkpoint = torch.load(join(name, "%s.pkl"%(model)))
        net.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print("loaded.")

        dirpath = join(distination, basename(normpath(name))+"_"+model)
        if not isdir(dirpath):
            os.mkdir(dirpath)

        proteins = np.load("/home/hiranumn/PyTorchDeepAccNet/data/test_proteins2.npy")
        for pname in proteins:
            print(pname)
            dirpath = join(distination, basename(normpath(name))+"_"+model, pname)
            if not isdir(dirpath):
                os.mkdir(dirpath)
            decoys = dan.DecoyDataset(targets = [pname], 
                                       lengthmax = lengthmax, 
                                       bert = bert,
                                       multi_dir = multi_dir,
                                       features = features)

            if pname in decoys.proteins:
                with torch.no_grad():
                    for i in range(len(decoys.samples_dict[pname])):
                        data = decoys.__getitem__(0, pindex=i)
                        idx, val, f1d, f2d, esto_1hot, esto, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                         data["estogram_1hot"], data["estogram"], data["mask"]

                        idx = torch.Tensor(idx).long().to(device)
                        val = torch.Tensor(val).to(device)
                        f1d = torch.Tensor(f1d).to(device)
                        f2d = torch.Tensor(f2d).to(device)
                        esto = torch.Tensor(esto).to(device)
                        esto_1hot = torch.Tensor(esto_1hot).to(device)
                        mask = torch.Tensor(mask).to(device)
                        lddt = dan.calculate_LDDT(esto_1hot[0], mask[0])

                        esto_pred, mask_pred, lddt_pred, dmy = net(idx, val, f1d, f2d)

                        samplename = basename(decoys.samples_dict[pname][i])
                        filepath = join(distination, basename(normpath(name))+"_"+model, pname, samplename+".npz")
                        np.savez_compressed(filepath,
                                            lddt = lddt_pred.cpu().detach().numpy(),
                                            estogram = esto_pred.cpu().detach().numpy(),
                                            mask = mask_pred.cpu().detach().numpy(),
                                            lddt_true = lddt.cpu().detach().numpy(),
                                            estogram_true = esto_1hot[0].cpu().detach().numpy(),
                                            mask_true = mask[0].cpu().detach().numpy())
    
if __name__== "__main__":
    main()