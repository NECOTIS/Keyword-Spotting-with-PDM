# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:22:17 2023

@author: asus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from paralif import ParaLIF



class ParaLIF_Net(nn.Module):
    def __init__(self, n_input, n_output, n_layers=2, n_hidden=128, spike_mode='D', rec=False, conv=False, k_size=3, 
                 device=None, tau_mem=1e-2, tau_syn=1e-2, dilation=1, delay=30, paralif_conv_groups=1):
        super().__init__()

        if (type(n_hidden) != list): n_hidden = [n_hidden]*n_layers
        if (type(rec) != list): rec = [rec]*n_layers
        if (type(conv) != list): conv = [conv]*n_layers
        if (type(k_size) != list): k_size = [k_size]*n_layers
        if (type(dilation) != list): dilation = [dilation]*n_layers
        if (type(delay) != list): delay = [delay]*n_layers
        if (type(paralif_conv_groups) != list): paralif_conv_groups = [1]+[paralif_conv_groups]*(n_layers-1)
        
        net_layers = []
        for i in range(n_layers):
            # Add the hidden ParaLif Layer
            net_layers.append(ParaLIF((n_input if i==0 else n_hidden[i-1]), n_hidden[i], spike_mode=spike_mode, 
                                      recurrent=rec[i], convolutional=conv[i], kernel_size=k_size[i], device=device, 
                                      dilation=dilation[i], groups=paralif_conv_groups[i], tau_mem=tau_mem, tau_syn=tau_syn))
            # Add the axonal delay
            if delay[i]>0: net_layers.append(DELAY(n_hidden[i], delay[i], device))
        # Add the output LI Layer
        net_layers.append(ParaLIF(n_hidden[-1], n_output, recurrent=False, convolutional=False, 
                        fire=False, device=device, tau_mem=tau_mem, tau_syn=tau_syn))
        self.net = torch.nn.Sequential(*net_layers)
        
                
    def forward(self, x):
        return self.net(x.permute(0,2,1))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Fixed axonal delay
class DELAY(nn.Module):
    def __init__(self, n_input, delay_max=10, device=None):
        super().__init__()
        self.register_buffer('delays', torch.randint(0, delay_max+1, (n_input,), device=device).to(torch.long))
        self.register_buffer('delay_max', torch.tensor(delay_max))
        self.device = device
    
    def roll(self, x, shifts):
        indices = (torch.arange(x.shape[0], device=self.device)[:, None] - shifts[None, :]) % x.shape[0]
        return torch.gather(x, 0, indices.long())

    def forward(self, x):
        x = F.pad(x, (0,0,0,self.delay_max), "constant", 0)
        return torch.stack([self.roll(x_i, self.delays) for x_i in x])
    
    
