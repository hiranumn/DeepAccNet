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

from pyrosetta import *
from pyrosetta.rosetta import *
init(extra_options = "-constant_seed -mute all")

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
        
    ##############################
    # Importing larger libraries #
    ##############################
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import deepAccNet as dan
    
    model = dan.DeepAccNet(twobody_size = 49 if args.bert else 33)
    checkpoint = torch.load(join(modelpath, "best.pkl"))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() or args.cpu else "cpu")
    model.to(device)
    model.eval()
    
    #############################
    # Parse through silent file #
    #############################
    
    silent_files = utility.vector1_utility_file_FileName()
    for silent_file in basic.options.get_file_vector_option("in:file:silent"):
        silent_files.append(utility.file.FileName(args.infile))
    input_stream = core.import_pose.pose_stream.SilentFilePoseInputStream(args.infile)

    # Open with append
    if not isfile(args.outfile) or args.reprocess:
        outfile = open(args.outfile, "w")
        outfile.write("name, global_lddt, interface_lddt, binder_lddt\n")
        done = []
    else:
        outfile = open(args.outfile, "a")
        done = pd.read_csv(args.outfile)["name"].values
    
    with torch.no_grad():
        # Parse through poses    
        pose = core.pose.Pose()
        while input_stream.has_another_pose():
            
            input_stream.fill_pose(pose)
            name = core.pose.tag_from_pose(pose)
            print(name)
            if name in done:
                continue
            per_sample_result = [name]
            
            # This is where featurization happens
            features = dan.process_from_pose(pose)
            
            # This is where prediction happens
            # For the whole 
            (idx, val), (f1d, bert), f2d, dmy = dan.getData_from_dict(features, bertpath = "")
            f1d_g = torch.Tensor(f1d).to(device)
            f2d_g = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
            idx_g = torch.Tensor(idx.astype(np.int32)).long().to(device)
            val_g = torch.Tensor(val).to(device)

            estogram, mask, lddt, dmy = model(idx_g, val_g, f1d_g, f2d_g)
            lddt = lddt.cpu().detach().numpy()
            estogram = estogram.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            
            # Store global lddt:
            per_sample_result.append(np.mean(lddt))
            
            # Binder related predictions
            if args.binder:
                
                # Binder length
                blen = pose.conformation().chain_end(1) - pose.conformation().chain_begin(1) + 1
                blen = 50
                plen = estogram.shape[-1]
                if blen==plen:
                    continue
                
                mask2 = np.zeros(mask.shape)
                mask2[:blen, blen:] = 1
                interface_lddt = np.mean(get_lddt(estogram.transpose([1,2,0]), np.multiply(mask, mask2)))
                per_sample_result.append(interface_lddt)
                
                # Subsample for binder prediction
                index = idx[:, 0] < blen
                idx = idx[index]
                val = val[index]
                idx_g = torch.Tensor(idx.astype(np.int32)).long().to(device)
                val_g = torch.Tensor(val).to(device)
                estogram, mask, lddt, dmy = model(idx_g, val_g, f1d_g[:blen], f2d_g[:, :, :blen, :blen])
                lddt = lddt.cpu().detach().numpy()
                estogram = estogram.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                per_sample_result.append(np.mean(lddt))
            
            # Write the result
            if args.binder:
                r = per_sample_result
                outfile.write("%s, %5f, %5f, %5f\n"%(r[0], r[1], r[2], r[3]))
            else:
                r = per_sample_result
                outfile.write("%s, %5f\n"%(r[0], r[1]))
            outfile.flush()
            os.fsync(outfile.fileno())
            
    outfile.close()
            
if __name__== "__main__":
    main()