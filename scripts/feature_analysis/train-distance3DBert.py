import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

import sys
sys.path.insert(0, "./")
import deepAccNet
    
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch

def main():
    parser = argparse.ArgumentParser(description="Error predictor network trainer",
                                     epilog="v0.0.1")
    
    parser.add_argument("folder",
                        action="store",
                        help="Location of folder to save checkpoints to.")
    
    parser.add_argument("--epoch",
                        "-e", action="store",
                        type=int,
                        default=200,
                        help="# of epochs (path over all proteins) to train for (Default: 200)")
    
    parser.add_argument("--bert",
                        "-bert",
                        action="store_true",
                        default=False,
                        help="Run with bert features (Default: False)")
    
    parser.add_argument("--multi_dir",
                        "-multi_dir",
                        action="store_true",
                        default=False,
                        help="Run with multiple direcotory sources (Default: False)")
    
    parser.add_argument("--num_blocks",
                        "-numb", action="store",
                        type=int,
                        default=5,
                        help="# of reidual blocks (Default: 8)")
    
    parser.add_argument("--num_filters",
                        "-numf", action="store",
                        type=int,
                        default=128,
                        help="# of base filter size in residual blocks (Default: 256)")
    
    parser.add_argument("--size_limit",
                        "-size_limit", action="store",
                        type=int,
                        default=280,
                        help="protein size limit (Default: 300)")
    
    parser.add_argument("--decay",
                        "-d", action="store",
                        type=float,
                        default=0.99,
                        help="Decay rate for learning rate (Default: 0.99)")
    
    parser.add_argument("--base",
                        "-b", action="store",
                        type=float,
                        default=0.0005,
                        help="Base learning rate (Default: 0.0005)")
    
    parser.add_argument("--debug",
                        "-debug",
                        action="store_true",
                        default=False,
                        help="Debug mode (Default: False)")
    
    parser.add_argument("--silent",
                        "-s",
                        action="store_true",
                        default=False,
                        help="Run in silent mode (Default: False)")
   
    args = parser.parse_args()
    
    script_dir = os.path.dirname(__file__)
    base = join(script_dir, "data/")

    epochs = args.epoch
    base_learning_rate = args.base
    decay = args.decay
    loss_weight = [1, 0.25, 10] #change if you need different loss
    validation = True #validation is always up
    name = args.folder
    lengthmax = args.size_limit
    
    if not args.silent: print("Loading samples")
        
    ###############
    # Change here 
    ###############
    features = ["distance", "distance2", "bert"]
        
    proteins = np.load(join(base, "train_proteins4.npy"))
    if args.debug: proteins = proteins[:100]
    train_decoys = deepAccNet.DecoyDataset(targets = proteins,
                                           lengthmax = lengthmax,
                                           bert = args.bert,
                                           multi_dir = args.multi_dir,
                                           features = features)
    train_dataloader = DataLoader(train_decoys, batch_size=1, shuffle=True, num_workers=4)

    proteins = np.load(join(base, "valid_proteins4.npy"))
    if args.debug: proteins = proteins[:100]
    valid_decoys = deepAccNet.DecoyDataset(targets = proteins, 
                                           lengthmax = lengthmax, 
                                           bert = args.bert,
                                           multi_dir = args.multi_dir,
                                           features = features)
    valid_dataloader = DataLoader(valid_decoys, batch_size=1, shuffle=True, num_workers=4)
    
    # Load the model if needed 
    if not args.silent: print("Instantitating a model")
        
    ###############
    # Change here 
    ###############
    net = deepAccNet.DeepAccNet_no1D(num_chunks   = args.num_blocks,
                                        num_channel  = args.num_filters,
                                        onebody_size = 0,
                                        twobody_size = 21)
    restoreModel = False
    
    if isdir(args.folder): 
        if not args.silent: print("Loading a checkpoint")
        checkpoint = torch.load(join(name, "model.pkl"))
        net.load_state_dict(checkpoint["model_state_dict"])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        best_models = checkpoint["best_models"]
        if not args.silent: print("Restarting at epoch", epoch)
        assert(len(train_loss["total"]) == epoch)
        assert(len(valid_loss["total"]) == epoch)
        restoreModel = True
    else:
        if not args.silent: print("Training a new model")
        epoch = 0
        train_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
        valid_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
        best_models = []
        if not isdir(name):
            if not args.silent: print("Creating a new dir at", name)
            os.mkdir(name)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    if restoreModel:
        checkpoint = torch.load(join(name, "model.pkl"))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Loop over the dataset multiple times
    start_epoch = epoch
    for epoch in range(start_epoch, epochs):  

        # Update the learning rate
        lr = base_learning_rate*np.power(decay, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Loop over batches
        net.train(True)
        temp_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
        for i, data in enumerate(train_dataloader):

            # Get the data, Hardcoded transformation for whatever reasons.
            idx, val, f1d, f2d, esto, esto_1hot, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                        data["estogram"], data["estogram_1hot"], data["mask"]
            idx = idx[0].long().to(device)
            val = val[0].to(device)
            f1d = f1d[0].to(device)
            f2d = f2d[0].to(device)
            esto_true = esto[0].to(device)
            esto_1hot_true = esto_1hot[0].to(device)
            mask_true = mask[0].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            esto_pred, mask_pred, lddt_pred, (esto_logits, mask_logits) = net(idx, val, f1d, f2d)
            lddt_true = deepAccNet.calculate_LDDT(esto_1hot_true[0], mask_true[0])

            Esto_Loss = torch.nn.CrossEntropyLoss()
            Mask_Loss = torch.nn.BCEWithLogitsLoss()
            Lddt_Loss = torch.nn.MSELoss()

            esto_loss = Esto_Loss(esto_logits, esto_true.long())
            mask_loss = Mask_Loss(mask_logits, mask_true)
            lddt_loss = Lddt_Loss(lddt_pred, lddt_true.float())

            loss = loss_weight[0]*esto_loss + loss_weight[1]*mask_loss + loss_weight[2]*lddt_loss
            loss.backward()
            optimizer.step()

            # Get training loss
            temp_loss["total"].append(loss.cpu().detach().numpy())
            temp_loss["esto"].append(esto_loss.cpu().detach().numpy())
            temp_loss["mask"].append(mask_loss.cpu().detach().numpy())
            temp_loss["lddt"].append(lddt_loss.cpu().detach().numpy())

            # Display training results
            sys.stdout.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f, esto-loss: %.2f, lddt-loss: %.2f, mask: %.2f"
                             %(epoch, epochs, i, len(train_decoys),
                               temp_loss["total"][-1], temp_loss["esto"][-1], temp_loss["lddt"][-1], temp_loss["mask"][-1]))

        train_loss["total"].append(np.array(temp_loss["total"]))
        train_loss["esto"].append(np.array(temp_loss["esto"]))
        train_loss["mask"].append(np.array(temp_loss["mask"]))
        train_loss["lddt"].append(np.array(temp_loss["lddt"]))

        if validation:
            net.eval() # turn off training mode
            temp_loss = {"total":[], "esto":[], "mask":[], "lddt":[]}
            with torch.no_grad(): # wihout tracking gradients
                for i, data in enumerate(valid_dataloader):

                    # Get the data, Hardcoded transformation for whatever reasons.
                    idx, val, f1d, f2d, esto, esto_1hot, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                                data["estogram"], data["estogram_1hot"], data["mask"]
                    idx = idx[0].long().to(device)
                    val = val[0].to(device)
                    f1d = f1d[0].to(device)
                    f2d = f2d[0].to(device)
                    esto_true = esto[0].to(device)
                    esto_1hot_true = esto_1hot[0].to(device)
                    mask_true = mask[0].to(device)

                    # forward + backward + optimize
                    esto_pred, mask_pred, lddt_pred, (esto_logits, mask_logits) = net(idx, val, f1d, f2d)
                    lddt_true = deepAccNet.calculate_LDDT(esto_1hot_true[0], mask_true[0])

                    Esto_Loss = torch.nn.CrossEntropyLoss()
                    Mask_Loss = torch.nn.BCEWithLogitsLoss()
                    Lddt_Loss = torch.nn.MSELoss()

                    esto_loss = Esto_Loss(esto_logits, esto_true.long())
                    mask_loss = Mask_Loss(mask_logits, mask_true)
                    lddt_loss = Lddt_Loss(lddt_pred, lddt_true.float())

                    loss = loss_weight[0]*esto_loss + loss_weight[1]*mask_loss + loss_weight[2]*lddt_loss

                    # Get training loss
                    temp_loss["total"].append(loss.cpu().detach().numpy())
                    temp_loss["esto"].append(esto_loss.cpu().detach().numpy())
                    temp_loss["mask"].append(mask_loss.cpu().detach().numpy())
                    temp_loss["lddt"].append(lddt_loss.cpu().detach().numpy())

            valid_loss["total"].append(np.array(temp_loss["total"]))
            valid_loss["esto"].append(np.array(temp_loss["esto"]))
            valid_loss["mask"].append(np.array(temp_loss["mask"]))
            valid_loss["lddt"].append(np.array(temp_loss["lddt"]))

            # Saving the model if needed.
            if name != "" and validation:
                
                folder = name
                # Name of ranked models. I know it is not optimal way to do it but the easiest fix is this.
                name_map = ["best.pkl", "second.pkl", "third.pkl", "fourth.pkl", "fifth.pkl"]
                
                new_model = (epoch, np.mean(valid_loss["total"][-1]))
                new_best_models = best_models[:]
                new_best_models.append(new_model)
                new_best_models.sort(key=lambda x:x[1])
                temp = new_best_models[:len(name_map)]
                new_best_models = [(temp[i][0], temp[i][1], name_map[i]) for i in range(len(temp))]

                # Saving and moving
                for i in range(len(new_best_models)):
                    m, performance, filename = new_best_models[i]
                    if m in [j[0] for j in best_models]:
                        index = [j[0] for j in best_models].index(m)
                        command = "mv %s %s"%(join(folder, best_models[index][2]), join(folder, "temp_"+new_best_models[i][2]))
                        os.system(command)
                    else:
                         torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                        }, join(folder, "temp_"+new_best_models[i][2]))

                # Renaming
                for i in range(len(new_best_models)):
                    command = "mv %s %s"%(join(folder, "temp_"+name_map[i]), join(folder, name_map[i]))
                    os.system(command)

                # Update best list                              
                best_models = new_best_models
            
            # Save all models
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'best_models' : best_models
                    }, join(name, "model.pkl"))

            # Saving progress plot
            for label in ["total", "esto", "mask", "lddt"]:

                width = 50
                # Train plot
                y_train = np.concatenate(train_loss[label])
                y_train_conved = np.convolve(y_train, np.ones((width,))/width, mode='valid')
                x_train = (np.arange(len(y_train))/len(train_decoys))[:-1*(width-1)]
                # Valid plot
                y_valid = np.concatenate(valid_loss[label])
                y_valid_conved = np.convolve(y_valid, np.ones((width,))/width, mode='valid')
                x_valid = (np.arange(len(y_valid))/len(valid_decoys))[:-1*(width-1)]

                plt.figure()
                plt.plot(x_train, y_train_conved, label="train")
                plt.plot(x_valid, y_valid_conved, label="valid")
                plt.legend(loc=1)
                plt.savefig(join(name, label+".png"))
                plt.close()

if __name__== "__main__":
    main()