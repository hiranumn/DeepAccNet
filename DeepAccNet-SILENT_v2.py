#!/software/conda/envs/tensorflow/bin/python
import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import torch
import time
import pandas as pd
import os
import glob

from pyrosetta import *
from pyrosetta.rosetta import *
init(extra_options = "-constant_seed -mute all -read_only_ATOM_entries")

def get_lddt(estogram, mask, center=7, weights=[1,1,1,1]):  
    # Remove diagonal from the mask.
    mask = np.multiply(mask, np.ones(mask.shape)-np.eye(mask.shape[0]))
    # Masking the estogram except for the last cahnnel
    masked = np.transpose(np.multiply(np.transpose(estogram, [2,0,1]), mask), [1,2,0])

    p0 = np.sum(masked[:,:,center], axis=-1)
    p1 = np.sum(masked[:,:,center-1]+masked[:,:,center+1], axis=-1)
    p2 = np.sum(masked[:,:,center-2]+masked[:,:,center+2], axis=-1)
    p3 = np.sum(masked[:,:,center-3]+masked[:,:,center+3], axis=-1)
    p4 = np.sum(mask, axis=-1)

    # Only work on parts where interaction happen
    output = np.divide((weights[0]*p0 + weights[1]*(p0+p1) + weights[2]*(p0+p1+p2) + weights[3]*(p0+p1+p2+p3))/np.sum(weights), p4, where=p4!=0)
    return output[p4!=0]

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Error predictor network",
                                     epilog="v0.0.1")
    parser.add_argument("infile",
                        action="store",
                        help="path to input silent file")
    
    parser.add_argument("outfile",
                        action="store",
                        help="path to output csv")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Activating verbose flag (Default: False)")
    
    parser.add_argument("--binder",
                        "-b",
                        action="store_true",
                        default=False,
                        help="Make binder related predictions (Assumes chain A to be a binder).")
    
    parser.add_argument("--savehidden",
                        "-sh", action="store",
                        type=str,
                        default="",
                        help="saves last hidden layer if not empty (Default: "")")
    
    parser.add_argument("--reprocess",
                        "-r",
                        action="store_true",
                        default=False,
                        help="Do not ignore already processed files")
    
    parser.add_argument("--bert",
                        "-bert",
                        action="store_true",
                        default=False,
                        help="Run with bert features. Use extractBert.py to generate them. (Default: False)")

    parser.add_argument("--features_only",
                        action="store_true",
                        help="Just dump features")

    parser.add_argument("--prediction_only",
                        action="store_true",
                        help="Assumes stored features")
    
    args = parser.parse_args()
    
    ################################
    # Checking file availabilities #
    ################################

    if not isfile(args.infile):
        print("Input silent file does not exist.", file=sys.stderr)
        return -1
        
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    
    if not args.bert:
        modelpath = join(base, "NatComm_standard")
    else:
        modelpath = join(base, "NatComm_bert")
    
    if not isdir(modelpath):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1
    
    if args.verbose: print("using", modelpath)

    feature_folder = args.outfile + "_features/"

    if ( args.features_only ):
        if ( not os.path.exists(feature_folder) ):
            os.mkdir(feature_folder)

    if ( args.prediction_only ):
        if ( not os.path.exists(feature_folder)):
            print("--prediction_only: Features have not been generated. Run with --features_only first or remove this flag.")
            return -1

    if ( args.features_only and args.prediction_only ):
        print("You can't specify both --features_only and --prediction_only at the same time.")
        return -1
        
    ##############################
    # Importing larger libraries #
    ##############################
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import deepAccNet as dan
    
    if ( not args.features_only ):
        model = dan.DeepAccNet(twobody_size = 49 if args.bert else 33)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(join(modelpath, "best.pkl"), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    
    #############################
    # Parse through silent file #
    #############################
    
    # loading the silent like this allows us to get names without loading poses
    sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
    sfd_in.read_file(args.infile)
    names = sfd_in.tags()

    # Open with append
    if not isfile(args.outfile) or args.reprocess:
        outfile = open(args.outfile, "w")
        if args.binder:
            outfile.write("global_lddt interface_lddt binder_lddt description\n")
        else:
            outfile.write("global_lddt description\n")
        done = []
    else:
        outfile = open(args.outfile, "a")
        done = pd.read_csv(args.outfile, sep="\s+")["description"].values

        
    if args.savehidden != "" and not isdir(args.savehidden):
        os.mkdir(args.savehidden)
    
    with torch.no_grad():
        # Parse through poses    
        pose = core.pose.Pose()
        for name in names:

            if name in done:
                print(name, "is already done.")
                continue


            print("Working on", name)
            per_sample_result = [name]
            feature_file = feature_folder + name


            
            # This is where featurization happens
            if ( args.prediction_only ):
                try:
                    features = np.load(feature_file + ".npz")
                except:
                    print("Unable to load features for " + name)
                    continue
            else:
                if ( args.features_only and os.path.exists( feature_file + ".npz" )):
                    print(name, "is already done.")
                    continue

                sfd_in.get_structure(name).fill_pose(pose)
                features = dan.process_from_pose(pose)
                features['blen'] = np.array(pose.conformation().chain_end(1) - pose.conformation().chain_begin(1) + 1)

            if ( args.features_only ):
                np.savez(feature_file, **features)
                continue

            
            # This is where prediction happens
            # For the whole 
            (idx, val), (f1d, bert), f2d, dmy = dan.getData_from_dict(features, bertpath = "")
            f1d_g = torch.Tensor(f1d).to(device)
            f2d_g = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
            idx_g = torch.Tensor(idx.astype(np.int32)).long().to(device)
            val_g = torch.Tensor(val).to(device)

            if args.savehidden != "":
                estogram, mask, lddt, hidden, dmy = model(idx_g, val_g, f1d_g, f2d_g, output_hidden_layer=True)
                hidden = hidden.cpu().detach().numpy()
                np.save(join(args.savehidden, name+".npy"), hidden)
            else:
                estogram, mask, lddt, dmy = model(idx_g, val_g, f1d_g, f2d_g)
            lddt = lddt.cpu().detach().numpy()
            estogram = estogram.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            
            # Store global lddt:
            per_sample_result.append(np.mean(lddt))
            
            # Binder related predictions
            if args.binder:
                
                # Binder length
                blen = features['blen']
                plen = estogram.shape[-1]
                if blen==plen:
                    continue
                
                mask2 = np.zeros(mask.shape)
                mask2[:blen, blen:] = 1
                mask2[blen:, :blen] = 1 
                interface_lddt = np.mean(get_lddt(estogram.transpose([1,2,0]), np.multiply(mask, mask2)))
                per_sample_result.append(interface_lddt)
                
                # Subsample for binder prediction
                index = idx[:, 0] < blen
                idx = idx[index]
                val = val[index]
                idx_g = torch.Tensor(idx.astype(np.int32)).long().to(device)
                val_g = torch.Tensor(val).to(device)
                if args.savehidden != "":
                    estogram, mask, lddt, hidden, dmy = model(idx_g, val_g, f1d_g[:blen], f2d_g[:, :, :blen, :blen], output_hidden_layer=True)
                    hidden = hidden.cpu().detach().numpy()
                    np.save(join(args.savehidden, name+"_b.npy"), hidden)
                else:
                    estogram, mask, lddt, dmy = model(idx_g, val_g, f1d_g[:blen], f2d_g[:, :, :blen, :blen])
                lddt = lddt.cpu().detach().numpy()
                estogram = estogram.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                per_sample_result.append(np.mean(lddt))
            
            # Write the result
            if args.binder:
                r = per_sample_result
                outfile.write("%5f %5f %5f %s\n"%(r[1], r[2], r[3], r[0]))
            else:
                r = per_sample_result
                outfile.write("%5f %s\n"%(r[1], r[0]))
            outfile.flush()
            os.fsync(outfile.fileno())

            if ( args.prediction_only ):
                os.remove(feature_file + ".npz")
            
    outfile.close()
            
if __name__== "__main__":
    main()
