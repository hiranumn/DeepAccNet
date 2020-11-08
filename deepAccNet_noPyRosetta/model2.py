# Feature testing models

import torch
import numpy as np
from torch.nn import functional as F
from .resnet import *

class DeepAccNet_no3D(torch.nn.Module):
    
    # Parameter initialization
    def __init__(self, 
                 onebody_size = 70,
                 twobody_size = 33,
                 protein_size = None,
                 num_chunks   = 5,
                 num_channel  = 128,
                 num_restype  = 20,
                 name = None,
                 loss_weight=[1.0, 0.25, 10.0],
                 verbose=False):
        
        self.onebody_size = onebody_size
        self.twobody_size = twobody_size
        self.protein_size = protein_size
        self.num_chunks   = num_chunks
        self.num_channel  = num_channel
        self.name = name
        self.loss_weight = loss_weight
        self.verbose = verbose
        
        super(DeepAccNet_no3D, self).__init__()
        
        # Transformation before ResNet
        self.add_module("conv1d_1", torch.nn.Conv1d(self.onebody_size, self.num_channel//2, 1, padding=0, bias=True))
        self.add_module("conv2d_1", torch.nn.Conv2d(self.num_channel+self.twobody_size, self.num_channel, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))
        
        # Build ResNet
        self.add_module("base_resnet", ResNet(num_channel,
                                              self.num_chunks,
                                              "base_resnet",
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))
        
        self.add_module("error_resnet", ResNet(num_channel,
                                               1,
                                               "error_resnet", 
                                               inorm=False,
                                               initial_projection=True,
                                               extra_blocks=True))
        self.add_module("conv2d_error", torch.nn.Conv2d(self.num_channel, 15, 1, padding=0, bias=True))
        
        self.add_module("mask_resnet", ResNet(num_channel,
                                              1,
                                              "mask_resnet",
                                              inorm=False,
                                              initial_projection=True,
                                              extra_blocks=True))
        self.add_module("conv2d_mask", torch.nn.Conv2d(self.num_channel, 1, 1, padding=0, bias=True))

        
    # Forward pass
    def forward(self, idx, val, obt, tbt):
        nres = obt.shape[0]
        
        # Permutation necesasry to match tf 
        out_cat_1 = torch.cat([obt], dim=1).unsqueeze(0).permute(0,2,1)
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](out_cat_1))
        
        # Do stacking with two body terms
        temp1 = tile(out_conv1d_1.unsqueeze(3), 3, nres)
        temp2 = tile(out_conv1d_1.unsqueeze(2), 2, nres)
        out_cat_2 = torch.cat([temp1, temp2, tbt], dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))
        
        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))
        
        # Error arm
        out_error_predictor = F.elu(self._modules["error_resnet"](out_base_resnet))
        estogram_logits = self._modules["conv2d_error"](out_error_predictor)
        estogram_logits = (estogram_logits + estogram_logits.permute(0,1,3,2))/2
        estogram_prediction = F.softmax(estogram_logits, dim=1)[0]
        
        # Mask arm
        out_mask_predictor = F.elu(self._modules["mask_resnet"](out_base_resnet))
        mask_logits = self._modules["conv2d_mask"](out_mask_predictor)[:,0,:,:] # Reduce the second dimension
        mask_logits = (mask_logits + mask_logits.permute(0,2,1))/2
        mask_prediction = torch.sigmoid(mask_logits)[0]
        
        # Calculate l-DDT
        lddt_prediction = calculate_LDDT(estogram_prediction, mask_prediction)
        
        return estogram_prediction, mask_prediction, lddt_prediction, (estogram_logits, mask_logits)
    
class DeepAccNet_no1D(torch.nn.Module):
    
    # Parameter initialization
    def __init__(self, 
                 onebody_size = 70,
                 twobody_size = 33,
                 protein_size = None,
                 num_chunks   = 5,
                 num_channel  = 128,
                 num_restype  = 20,
                 name = None,
                 loss_weight=[1.0, 0.25, 10.0],
                 verbose=False):
        
        self.twobody_size = twobody_size
        self.protein_size = protein_size
        self.num_chunks   = num_chunks
        self.num_channel  = num_channel
        self.num_restype  = num_restype
        self.name = name
        self.loss_weight = loss_weight
        self.verbose = verbose
        
        super(DeepAccNet_no1D, self).__init__()
        
        # 3D Convolutions without padding.
        self.add_module("retype", torch.nn.Conv3d(self.num_restype, 20, 1, padding=0, bias=False))
        self.add_module("conv3d_1", torch.nn.Conv3d(20, 20, 3, padding=0, bias=True))
        self.add_module("conv3d_2", torch.nn.Conv3d(20, 30, 4, padding=0, bias=True))
        self.add_module("conv3d_3", torch.nn.Conv3d(30, 10, 4, padding=0, bias=True))
        self.add_module("pool3d_1", torch.nn.AvgPool3d(kernel_size=4, stride=4, padding=0))
        
        # Transformation before ResNet
        self.add_module("conv1d_1", torch.nn.Conv1d(640, self.num_channel//2, 1, padding=0, bias=True))
        self.add_module("conv2d_1", torch.nn.Conv2d(self.num_channel+self.twobody_size, self.num_channel, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))
        
        # Build ResNet
        self.add_module("base_resnet", ResNet(num_channel,
                                              self.num_chunks,
                                              "base_resnet",
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))
        
        self.add_module("error_resnet", ResNet(num_channel,
                                               1,
                                               "error_resnet", 
                                               inorm=False,
                                               initial_projection=True,
                                               extra_blocks=True))
        self.add_module("conv2d_error", torch.nn.Conv2d(self.num_channel, 15, 1, padding=0, bias=True))
        
        self.add_module("mask_resnet", ResNet(num_channel,
                                              1,
                                              "mask_resnet",
                                              inorm=False,
                                              initial_projection=True,
                                              extra_blocks=True))
        self.add_module("conv2d_mask", torch.nn.Conv2d(self.num_channel, 1, 1, padding=0, bias=True))

        
    # Forward pass
    def forward(self, idx, val, obt, tbt):
        nres = tbt.shape[-1]
        
        # Grid3D with custom scatter_nd
        x = scatter_nd(idx, val, (nres, 24, 24, 24, self.num_restype))
        x = x.permute(0,4,1,2,3)
        
        # 3Dconvolution 
        out_retype = self._modules["retype"](x)
        out_conv3d_1 = F.elu(self._modules["conv3d_1"](out_retype))
        out_conv3d_2 = F.elu(self._modules["conv3d_2"](out_conv3d_1))
        out_conv3d_3 = F.elu(self._modules["conv3d_3"](out_conv3d_2))
        out_pool3d_1 = self._modules["pool3d_1"](out_conv3d_3)
        
        # Permutation necesasry to match tf 
        flattened_1 = torch.flatten(out_pool3d_1.permute(0,2,3,4,1), start_dim=1, end_dim=-1)
        out_cat_1 = torch.cat([flattened_1], dim=1).unsqueeze(0).permute(0,2,1)
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](out_cat_1))
        
        # Do stacking with two body terms
        temp1 = tile(out_conv1d_1.unsqueeze(3), 3, nres)
        temp2 = tile(out_conv1d_1.unsqueeze(2), 2, nres)
        out_cat_2 = torch.cat([temp1, temp2, tbt], dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))
        
        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))
        
        # Error arm
        out_error_predictor = F.elu(self._modules["error_resnet"](out_base_resnet))
        estogram_logits = self._modules["conv2d_error"](out_error_predictor)
        estogram_logits = (estogram_logits + estogram_logits.permute(0,1,3,2))/2
        estogram_prediction = F.softmax(estogram_logits, dim=1)[0]
        
        # Mask arm
        out_mask_predictor = F.elu(self._modules["mask_resnet"](out_base_resnet))
        mask_logits = self._modules["conv2d_mask"](out_mask_predictor)[:,0,:,:] # Reduce the second dimension
        mask_logits = (mask_logits + mask_logits.permute(0,2,1))/2
        mask_prediction = torch.sigmoid(mask_logits)[0]
        
        # Calculate l-DDT
        lddt_prediction = calculate_LDDT(estogram_prediction, mask_prediction)
        
        return estogram_prediction, mask_prediction, lddt_prediction, (estogram_logits, mask_logits)
    
class DeepAccNet_no3Dno1D(torch.nn.Module):
    
    # Parameter initialization
    def __init__(self, 
                 onebody_size = 70,
                 twobody_size = 33,
                 protein_size = None,
                 num_chunks   = 5,
                 num_channel  = 128,
                 num_restype  = 20,
                 name = None,
                 loss_weight=[1.0, 0.25, 10.0],
                 verbose=False):
        
        self.twobody_size = twobody_size
        self.protein_size = protein_size
        self.num_chunks   = num_chunks
        self.num_channel  = num_channel
        self.name = name
        self.loss_weight = loss_weight
        self.verbose = verbose
        
        super(DeepAccNet_no3Dno1D, self).__init__()
        
        self.add_module("conv2d_1", torch.nn.Conv2d(self.twobody_size, self.num_channel, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))
        
        # Build ResNet
        self.add_module("base_resnet", ResNet(num_channel,
                                              self.num_chunks,
                                              "base_resnet",
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))
        
        self.add_module("error_resnet", ResNet(num_channel,
                                               1,
                                               "error_resnet", 
                                               inorm=False,
                                               initial_projection=True,
                                               extra_blocks=True))
        self.add_module("conv2d_error", torch.nn.Conv2d(self.num_channel, 15, 1, padding=0, bias=True))
        
        self.add_module("mask_resnet", ResNet(num_channel,
                                              1,
                                              "mask_resnet",
                                              inorm=False,
                                              initial_projection=True,
                                              extra_blocks=True))
        self.add_module("conv2d_mask", torch.nn.Conv2d(self.num_channel, 1, 1, padding=0, bias=True))

        
    # Forward pass
    def forward(self, idx, val, obt, tbt):
        nres = tbt.shape[-1]

        out_cat_2 = torch.cat([tbt], dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))
        
        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))
        
        # Error arm
        out_error_predictor = F.elu(self._modules["error_resnet"](out_base_resnet))
        estogram_logits = self._modules["conv2d_error"](out_error_predictor)
        estogram_logits = (estogram_logits + estogram_logits.permute(0,1,3,2))/2
        estogram_prediction = F.softmax(estogram_logits, dim=1)[0]
        
        # Mask arm
        out_mask_predictor = F.elu(self._modules["mask_resnet"](out_base_resnet))
        mask_logits = self._modules["conv2d_mask"](out_mask_predictor)[:,0,:,:] # Reduce the second dimension
        mask_logits = (mask_logits + mask_logits.permute(0,2,1))/2
        mask_prediction = torch.sigmoid(mask_logits)[0]
        
        # Calculate l-DDT
        lddt_prediction = calculate_LDDT(estogram_prediction, mask_prediction)
        
        return estogram_prediction, mask_prediction, lddt_prediction, (estogram_logits, mask_logits)
    
# Calculates LDDT based on estogram
def calculate_LDDT(estogram, mask, center=7):
    # Get on the same device as indices
    device = estogram.device

    # Remove diagonal from calculation
    nres = mask.shape[-1]
    mask = torch.mul(mask, torch.ones((nres, nres)).to(device) - torch.eye(nres).to(device))
    masked = torch.mul(estogram, mask)

    p0 = (masked[center]).sum(axis=0)
    p1 = (masked[center-1]+masked[center+1]).sum(axis=0)
    p2 = (masked[center-2]+masked[center+2]).sum(axis=0)
    p3 = (masked[center-3]+masked[center+3]).sum(axis=0)
    p4 = mask.sum(axis=0)

    return 0.25 * (4.0*p0 + 3.0*p1 + 2.0*p2 + p3) / p4
    
def tile(a, dim, n_tile):
    # Get on the same device as indices
    device = a.device
    
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)


# tf.scatter_nd-like function implemented with torch.scatter_add 
def scatter_nd(indices, updates, shape):
    # Get on the same device as indices
    device = indices.device
    
    # Initialize empty array
    size = np.prod(shape)
    out = torch.zeros(size).to(device)
    
    # Get flattened index (Calculation needs to be done in long to preserve indexing precision)
    temp = np.array([int(np.prod(shape[i+1:])) for i in range(len(shape))])
    flattened_indices = torch.mul(indices.long(), torch.as_tensor(temp, dtype=torch.long).to(device)).sum(dim=1)
    
    # Scatter_add
    out = out.scatter_add(0, flattened_indices, updates)
    
    # Reshape
    return out.view(shape)