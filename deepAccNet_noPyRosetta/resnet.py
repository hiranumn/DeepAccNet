import torch
import numpy as np
from torch.nn import functional as F

# Implementation of ResNet module

class ResNet(torch.nn.Module):
    
    # Parameter initialization
    def __init__(self, 
                 num_channel,
                 num_chunks,
                 name,
                 inorm              = False,
                 initial_projection = False,
                 extra_blocks       = False,
                 dilation_cycle     = [1,2,4,8],
                 verbose=False):
        
        self.num_channel = num_channel
        self.num_chunks = num_chunks
        self.name = name
        self.inorm = inorm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = dilation_cycle
        self.verbose = verbose
        
        super(ResNet, self).__init__()
        
        if self.initial_projection:
            self.add_module("resnet_%s_init_proj"%(self.name), torch.nn.Conv2d(num_channel, num_channel, 1))
            
        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:
                    self.add_module("resnet_%s_%i_%i_inorm_1"%(self.name, i, dilation_rate),
                                       torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_%i_%i_inorm_2"%(self.name, i, dilation_rate),
                                       torch.nn.InstanceNorm2d(num_channel//2, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_%i_%i_inorm_3"%(self.name, i, dilation_rate),
                                       torch.nn.InstanceNorm2d(num_channel//2, eps=1e-06, affine=True))

                self.add_module("resnet_%s_%i_%i_conv2d_1"%(self.name, i, dilation_rate),
                                    torch.nn.Conv2d(num_channel, num_channel//2, 1))
                self.add_module("resnet_%s_%i_%i_conv2d_2"%(self.name, i, dilation_rate),
                                    torch.nn.Conv2d(num_channel//2, num_channel//2, 3, dilation=dilation_rate, padding=dilation_rate))
                self.add_module("resnet_%s_%i_%i_conv2d_3"%(self.name, i, dilation_rate),
                                    torch.nn.Conv2d(num_channel//2, num_channel, 1))
                
        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module("resnet_%s_extra%i_inorm_1"%(self.name, i),
                                       torch.nn.InstanceNorm2d(num_channel, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_extra%i_inorm_2"%(self.name, i),
                                       torch.nn.InstanceNorm2d(num_channel//2, eps=1e-06, affine=True))
                    self.add_module("resnet_%s_extra%i_inorm_3"%(self.name, i),
                                       torch.nn.InstanceNorm2d(num_channel//2, eps=1e-06, affine=True))

                self.add_module("resnet_%s_extra%i_conv2d_1"%(self.name, i),
                                    torch.nn.Conv2d(num_channel, num_channel//2, 1))
                self.add_module("resnet_%s_extra%i_conv2d_2"%(self.name, i),
                                    torch.nn.Conv2d(num_channel//2, num_channel//2, 3, dilation=1, padding=1))
                self.add_module("resnet_%s_extra%i_conv2d_3"%(self.name, i),
                                    torch.nn.Conv2d(num_channel//2, num_channel, 1))

    def forward(self, x):
        
        if self.initial_projection:
            x = self._modules["resnet_%s_init_proj"%(self.name)](x)
            
        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x
                
                # Internal block
                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_1"%(self.name, i, dilation_rate)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_1"%(self.name, i, dilation_rate)](x)
                
                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_2"%(self.name, i, dilation_rate)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_2"%(self.name, i, dilation_rate)](x)
                
                if self.inorm: x = self._modules["resnet_%s_%i_%i_inorm_3"%(self.name, i, dilation_rate)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_%i_%i_conv2d_3"%(self.name, i, dilation_rate)](x)
                
                x = x+_residual
                
        if self.extra_blocks:
            for i in range(2):
                _residual = x
                
                # Internal block
                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_1"%(self.name, i)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_1"%(self.name, i)](x)
                
                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_2"%(self.name, i)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_2"%(self.name, i)](x)
                
                if self.inorm: x = self._modules["resnet_%s_extra%i_inorm_3"%(self.name, i)](x)
                x = F.elu(x)
                x = self._modules["resnet_%s_extra%i_conv2d_3"%(self.name, i)](x)
                
                x = x+_residual
                
        return x