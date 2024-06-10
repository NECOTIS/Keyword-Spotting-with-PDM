import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np


class PDMEncodeur_seq(torch.nn.Module):
    def __init__(self, pdm_factor=10, orig_freq=16000):
        super().__init__()
        self.pdm_factor = pdm_factor
        self.upsampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=orig_freq*pdm_factor)
        
    def to(self, dest):
        super().to(dest)
        self.upsampler = self.upsampler.to(dest)
        return self
    
    def forward(self, waveform):
        waveform = (waveform/2)+0.5
        if self.pdm_factor !=1: waveform = self.upsampler(waveform)
        spikes = torch.zeros_like(waveform)
        error = torch.zeros_like(waveform[:,0])
        for i in range(waveform.shape[1]):
            error += waveform[:,i]
            spikes[:,i] = error>0
            error -= spikes[:,i]
        return spikes


class PDMEncodeur(torch.nn.Module):
    def __init__(self, pdm_factor=10, orig_freq=16000):
        super().__init__()
        self.pdm_factor = pdm_factor
        self.upsampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=orig_freq*pdm_factor)
        self.th = 1.
        
    def to(self, dest):
        super().to(dest)
        self.upsampler = self.upsampler.to(dest)
        return self
    
    def forward(self, waveform):
        waveform = (waveform/2)+0.5
        if self.pdm_factor !=1: waveform = self.upsampler(waveform)
        spikes = torch.zeros_like(waveform)
        waveform = waveform.to(torch.float64)
        waveform_cumsum = torch.cumsum(waveform, dim=1)
        waveform_div = waveform_cumsum//self.th
        waveform_div_diff = waveform_div-F.pad(waveform_div[:,:-1], (1,0), value=-1)
        spikes[waveform_div_diff>0] = 1.
        return spikes


class DataAug(torch.nn.Module):
    def __init__(self, shift_factor=0.1, sample_rate=16000):
        super().__init__()
        self.shift_factor = int(shift_factor*sample_rate)

    def shift(self, waveform):
        shift_factor = random.randint(-self.shift_factor, self.shift_factor)
        if shift_factor>0: return F.pad(waveform[:,:-shift_factor], (shift_factor,0))
        elif shift_factor<0: return F.pad(waveform[:,-shift_factor:], (0,-shift_factor))
        else: return waveform

    def forward(self, waveform):
        waveform = self.shift(waveform)
        return waveform


"""
For all subsets (train, valid, test) the individuals waveforms are loaded with torchaudio, stacked and saved as a pytorch tensor (train_data.pt, valid_data.pt, test_data.pt)
For all subsets (train, valid, test) the corresponding labels are stacked and saved as numpy arrays (train_targets.npy, valid_targets.npy, test_targets.npy)
"""
class InMemoryGSCDataset(Dataset):
    def __init__(self, subset="train", root="./Data/", transform=None, n_examples=None, pdm_factor=10, device=None, **kwargs):
        super().__init__()
        self.labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 
                        'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

        self.sc_sample_rate = 16000
        self.sc_duration_s = 1
        self.device = device
        self.pdm_factor = pdm_factor
        self.pdm_encodeur = None
        self.transform = transform
        self.n_examples = n_examples
        
        self.waveforms = torch.load(root+f"SpeechCommands/{subset}_data.pt")
        labels_ = np.load(root+f"SpeechCommands/{subset}_targets.npy")
        self.label_indexes = torch.tensor([self.labels.index(l) for l in labels_])
        
        if self.pdm_factor>0: 
            self.pdm_encodeur = PDMEncodeur(self.pdm_factor, self.sc_sample_rate).to(device)

        if self.n_examples and self.n_examples<1 and self.n_examples>0: self.reduce_size()

        self.label_indexes = self.label_indexes.to(device)
    
    def reduce_size(self):
        part_len = int(self.n_examples*len(self.label_indexes))
        ind = np.arange(len(self.label_indexes))
        rng = np.random.default_rng(0)
        rng.shuffle(ind)
        ind = ind[:part_len]
        self.waveforms =  torch.stack([self.waveforms[i] for i in ind])
        self.label_indexes = torch.stack([self.label_indexes[i] for i in ind])
        
        
    def to(self, device):
        self.device = device
        if self.transform: self.transform = self.transform.to(device)
        if self.pdm_encodeur: self.pdm_encodeur = self.pdm_encodeur.to(device)
        if hasattr(self, 'waveforms'): self.waveforms = self.waveforms.to(device)
        if hasattr(self, 'label_indexes'): self.label_indexes = self.label_indexes.to(device)
        return self

    def __getitem__(self, n):
        waveform = self.waveforms[n]
        waveform = waveform.to(self.device)
        
        if self.transform:
           waveform = self.transform(waveform)
        if self.pdm_encodeur:
            waveform = self.pdm_encodeur(waveform)

        return waveform, self.label_indexes[n]
    
    def __len__(self):
        return len(self.label_indexes)



if __name__=='__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="./Data/")
    parser.add_argument("--transform", type=str, default=None)
    parser.add_argument("--pdm_factor", type=int, default=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    dataset = InMemoryGSCDataset(root=args.data_dir, subset="test",
                           transform=args.transform, pdm_factor=args.pdm_factor,
                           n_examples=args.n_examples, device=device)
    len(dataset)
    len(dataset.labels)
    class_weights = torch.unique(dataset.label_indexes, return_counts=True)[1]/len(dataset)
    sample_weights = class_weights[dataset.label_indexes].tolist()
    def plot_waveform(x, ax):
        ax.plot(x.cpu().squeeze())
        ax.set_xlabel('Sample Index')

    import matplotlib.pylab as plt
    for example_index in torch.randint(len(dataset), size=(2,)):
        waveform, index = dataset[example_index]
        
        fig, ax = plt.subplots(1, 1)
        plot_waveform(waveform, ax)
        #ax.set_title(dataset.labels[index])
        plt.show()