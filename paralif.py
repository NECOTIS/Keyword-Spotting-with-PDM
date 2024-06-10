# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F



class ParaLIF(nn.Module):
    def __init__(self, input_size, hidden_size, spike_mode="D", device=None, recurrent=True, convolutional=False, kernel_size=7,
                 fire=True, tau_mem=1e-2, tau_syn=1e-2, time_step=1e-3, dilation=1, groups=1):
        super(ParaLIF, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.recurrent = recurrent
        self.convolutional = convolutional
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.fire = fire
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        alpha = torch.exp(torch.tensor(-time_step/tau_syn)) if tau_syn!=0 else torch.tensor(0.)
        beta = torch.exp(torch.tensor(-time_step/tau_mem)) if tau_mem!=0 else torch.tensor(0.)
        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)
        self.register_buffer('beta_1', 1-beta)
        
        
        if self.convolutional:
            # 1D convolution layer for input synapses
            self.conv = torch.nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, stride=(self.kernel_size*self.dilation)//2, 
                                        dilation=self.dilation, device=self.device, groups=self.groups)
            # Initializing weights
            torch.nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='conv1d')
            torch.nn.init.zeros_(self.conv.bias)
        else:
            # Fully connected layer for input synapses
            self.fc = torch.nn.Linear(self.input_size, self.hidden_size, device=self.device)
            # Initializing weights
            torch.nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(self.fc.bias)
        # Fully connected for recurrent synapses 
        if self.recurrent:
            self.fc_recu = torch.nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
            # Initializing weights
            torch.nn.init.kaiming_uniform_(self.fc_recu.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(self.fc_recu.bias)
            
        # Set the spiking function
        self.spike_mode = spike_mode
        if self.fire: 
            self.spike_fn = SpikingFunction(self.device, self.spike_mode, self.hidden_size)
        self.nb_steps = None
        self.mean_spike_rate = torch.tensor(0., device=self.device)
        

    def compute_params_fft(self, device=None):
        """
        Compute the FFT of the parameters for parallel Leaky Integration

        Returns:
        fft_l_k: Product of FFT of parameters l and k
        """
        if self.nb_steps is None: return None
        if self.alpha==0: #current-based synapse
            fft_l=1 
        else:             #conductance-based synapse
            l = torch.pow(self.alpha,torch.arange(self.nb_steps,device=device))
            fft_l = torch.fft.rfft(l, n=2*self.nb_steps).unsqueeze(1)
        k = torch.pow(self.beta,torch.arange(self.nb_steps,device=device))*self.beta_1
        fft_k = torch.fft.rfft(k, n=2*self.nb_steps).unsqueeze(1)
        return fft_l*fft_k
    
    def forward(self, inputs):
        """
        Perform forward pass of the network

        Parameters:
        - inputs (tensor): Input tensor with shape (batch_size, nb_steps, input_size)

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is True
        """

        if self.convolutional: 
            X = self.conv(inputs.permute(0,2,1)).permute(0,2,1)
        else: X = self.fc(inputs)
        batch_size,nb_steps,_ = X.shape

        # Compute FFT params if nb_steps has changed
        if self.nb_steps != nb_steps: 
            self.nb_steps = nb_steps
            self.fft_l_k = self.compute_params_fft(X.device)

        # Perform parallel leaky integration
        fft_X = torch.fft.rfft(X, n=2*nb_steps, dim=1)
        mem_rec = torch.fft.irfft(fft_X*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
 
        
        if self.recurrent:
            mem_rec_croped = F.pad(mem_rec, (0,0,1,0), "constant", 0)[:,:-1]
            #spk_recur = self.spike_fn(mem_rec_croped)  #Original ParaLIF
            spk_recur = F.relu(mem_rec_croped) #Modified ParaLIF
            fft_recur = torch.fft.rfft(X + self.fc_recu(spk_recur), n=2*nb_steps, dim=1)
            mem_rec = mem_rec + torch.fft.irfft(fft_recur*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
        
        if not self.fire: return mem_rec
        spk_rec = self.spike_fn(mem_rec)
        self.mean_spike_rate = torch.mean(spk_rec,dim=0).sum()
        return spk_rec
    
    
    def extra_repr(self):
        return f"spike_mode={self.spike_mode}, recurrent={self.recurrent}, convolutional={self.convolutional}, fire={self.fire}, tau_mem={self.tau_mem}, tau_syn={self.tau_syn}"




    
class SpikingFunction(nn.Module):
    def __init__(self, device, spike_mode, n_hidden):
        super(SpikingFunction, self).__init__()
            
        if spike_mode in ["SB", "SD", "ST"]: self.normalise = torch.sigmoid
        elif spike_mode in ["TD", "TT"]: self.normalise = torch.tanh
        elif spike_mode in ["TRB", "TRD", "TRT"]: self.normalise = lambda inputs : F.relu(torch.tanh(inputs))
        else: self.normalise = lambda inputs : inputs
        
        if spike_mode in ["SB", "TRB"]: self.generate = StochasticStraightThrough.apply
        elif spike_mode =="GS": self.generate = GumbelSoftmax(device)
        elif spike_mode in ["D", "SD", "TD", "TRD"]: 
            self.generate = self.delta_fn
            self.register_buffer('threshold', 0.1*torch.rand(n_hidden, device=device)) # different threshold for each neuron
        elif spike_mode in ["T", "ST", "TT", "TRT"]: 
            self.generate = self.threshold_fn
            self.threshold = torch.nn.Parameter(self.normalise(torch.tensor(1., device=device)))
        
    def forward(self, inputs):
        inputs = self.normalise(inputs) 
        return self.generate(inputs)

    def delta_fn(self, inputs):
        inputs_offset = F.pad(inputs, (0,0,1,0), "constant", 0)[:,:-1]
        return SurrGradSpike.apply((inputs - inputs_offset) - self.threshold)

    def threshold_fn(self, inputs):
        return SurrGradSpike.apply(inputs - self.threshold)

class StochasticStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.bernoulli(input) # Equation (18)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input*input # Equation (19)

# Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
class SurrGradSpike(torch.autograd.Function):
    scale = 50.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad



class GumbelSoftmax(torch.nn.Module):
    def __init__(self, device, hard=True, tau=1.0):
        super().__init__()
        
        self.hard = hard
        self.tau = tau
        self.uniform = torch.distributions.Uniform(torch.tensor(0.0).to(device),
                                                   torch.tensor(1.0).to(device))
        self.softmax = torch.nn.Softmax(dim=0)
  
    def forward(self, logits):
        # Sample uniform noise
        unif = self.uniform.sample(logits.shape + (2,))
        # Compute Gumbel noise from the uniform noise
        gumbels = -torch.log(-torch.log(unif))
        # Apply softmax function to the logits and Gumbel noise
        y_soft = self.softmax(torch.stack([(logits + gumbels[...,0]) / self.tau,
                                                     (-logits + gumbels[...,1]) / self.tau]))[0]
        if self.hard:
            # Use straight-through estimator
            y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Use reparameterization trick
            ret = y_soft
        return ret