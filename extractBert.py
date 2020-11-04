from os import listdir
from os.path import join, isdir, isfile
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM
import glob
import time
import re

def parsePDB(filename, atom="CA"):
    file = open(filename, "r")
    lines = file.readlines()
    coords = []
    aas = []
    
    cur_resdex = -1
    aa = ""
    for line in lines:
        if "ATOM" in line:
            if cur_resdex != int(line[22:26]):
                cur_resdex = int(line[22:26])
                new_res = True
                aa = line[17:20]
                aas.append(aa)
            if atom == "CA" and " CA " == line[12:16]:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                coords.append(xyz)
            elif atom == "CB":
                if aa == "GLY" and " CA " == line[12:16]:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    coords.append(xyz)
                elif " CB " == line[12:16]:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    coords.append(xyz)
    return np.array(coords), aas

####################
# INDEXERS/MAPPERS 
####################
# Assigning numbers to 3 letter amino acids.
residues= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(residues[i], i) for i in range(len(residues))])

# Mapping 3 letter AA to 1 letter AA (e.g. ALA to A)
oneletter = ["A", "R", "N", "D", "C", \
 "Q", "E", "G", "H", "I", \
 "L", "K", "M", "F", "P", \
 "S", "T", "W", "Y", "V"]
aanamemap = dict([(residues[i], oneletter[i]) for i in range(len(residues))])

def parse_fasta(filename,limit=-1):
    '''function to parse fasta'''
    header = []
    sequence = []
    lines = open(filename, "r")
    for line in lines:
        line = line.rstrip()
        if line[0] == ">":
            if len(header) == limit:
                break
            header.append(line[1:])
            sequence.append([])
        else:
            sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return np.array(header), np.array(sequence)

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="ProtBert embedding generator",
                                     epilog="v0.0.1")
    parser.add_argument("input",
                        action="store",
                        help="path to input folder")
    
    parser.add_argument("output",
                        action="store",
                        help="path to output folder")
    
    parser.add_argument("--modelpath",
                        "-modelpath",
                        action="store",
                        default='/home/justas/Desktop/my_projects/python_runs/models/ProtBert-BFD/',
                        help="modelpath (default: /home/justas/Desktop/my_projects/python_runs/models/ProtBert-BFD/")
    
    args = parser.parse_args()
    
    if not isdir(args.output):
        os.mkdir(args.output)
        
    pdbfiles = [i for i in listdir(args.input) if i.endswith(".pdb")]

    for pdbfile in pdbfiles:
        try:
            coords, aas = parsePDB(join(args.input, pdbfile))
            output = ">"+pdbfile[:-4]+"\n"
            output += "".join([aanamemap[i] for i in aas])+"\n"
            f = open(join(args.output, pdbfile[:-4]+".fa"), "w")
            f.write(output)
            f.close()
        except:
            print(pdbfile)
        
        
    downloadFolderPath = args.modelpath
    modelFolderPath = downloadFolderPath
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )

    model = BertForMaskedLM.from_pretrained(modelFolderPath, output_attentions=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
        
    file_list = glob.glob(join(OUTPUT_PATH, "*.fa"))
    protein_names = []
    for i in file_list:
        name_1 = i.split("/")[-1]
        protein_names.append(name_1[:-3])

    start = time.time()
    for i in range(len(protein_names)):
        if i%100==0:
            print(100*(i+1)/len(protein_names))
        a, b = parse_fasta(join(OUTPUT_PATH, f"{protein_names[i]}.fa"))
        sequences_Example = [b[0].replace("", " ")[1: -1]]
        sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
        ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            Z_out= model(input_ids=input_ids, attention_mask=attention_mask)

        last_layer_attn = np.array((Z_out[1][-1].cpu().detach().numpy())[0,:,1:-1,1:-1], np.float32)

        np.save(join(OUTPUT_PATH, f'bert_{protein_names[i]}.npy'), last_layer_attn)
    print(f'total runtime: {time.time()-start} seconds')
    
    
if __name__== "__main__":
    main()