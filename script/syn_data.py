
# toy data generation

from torch.utils.data import DataLoader, Dataset, TensorDataset
from copy import deepcopy as copy
from scipy.io import loadmat, savemat

import torch, os 
import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np        
from math import pi, sqrt

class syn_multiview_dataset(Dataset):
    def __init__(self, 
                 root_dir:str="Data", 
                 name:str='synthetic',
                 freqs: list=[1.0,2.0,sqrt(5.0),pi],
                 bounds:list[float,float]=[-1,1],
                 step:float=0.001,
                 add_noise_weight:float=0.01,
                 fea_noise_weight:float=0.02,
                 noise_freq: float=3.6,
                 project_dim:int=20,
                 cat_noise: bool=True,
                 **kwargs):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir=root_dir
        self.name = name  

        self.figures = []
        self.bounds=bounds
        self.step=step
        self.add_noise_weight =add_noise_weight
        self.fea_noise_weight =fea_noise_weight
        self.cuda=kwargs['cuda']
        self.t = torch.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/step)+1).unsqueeze(0)*pi
        self.n_view = len(freqs)-1
        self.feature_dims = [project_dim for _ in range(self.n_view)]
        
        os.makedirs(os.path.join(root_dir, name), exist_ok=True)
        file_name=os.path.join(root_dir, name, f'{self.n_view}_view_sinusoidal_signals.mat')
        if os.path.exists(file_name):
            data = loadmat(file_name)
            self.com = torch.from_numpy(data['com'])
            self.unis = [torch.from_numpy(uni) for uni in data['unis']]
        else:
            self.com = torch.sin(freqs[0]*self.t)
            self.unis = [torch.cos(freq*self.t) for freq in freqs[1:]]
            savemat(file_name, dict(com=self.com.numpy(),
                                    unis=[uni.numpy() for uni in self.unis]))

        self.noise = fea_noise_weight*torch.sin(noise_freq*self.t)
        self.grouds, self.cleans, self.views_, self.maps = self.transform()

        if cat_noise:
            self.views_ = [torch.cat([view, self.noise]).T for view in self.views_]
            self.feature_dims = [dim+1 for dim in self.feature_dims]
        self.views = TensorDataset(*(self.views_))
        self.length = len(self.t.T)
        self.n_class= 1
        

    def transform(self,):
        ground, clean, noised, maps = [], [], [], []
        for i,uni in enumerate(self.unis):  
            # W = torch.linalg.svd(torch.randn((self.feature_dims[i], 2)))[-1]
            # W = W.T * (1 / torch.norm(W, dim=1, keepdim=True))
            W = torch.rand((self.feature_dims[i],2), device=uni.device)
            V = torch.cat([self.com, uni])
            clean_V = W.matmul(V)
            noise_V = clean_V + torch.randn_like(clean_V)*self.add_noise_weight            
            noise_V = 2*(noise_V - noise_V.min()) / (noise_V.max() - noise_V.min())- 1

            ground.append(V)
            maps.append(W)
            clean.append(clean_V)
            noised.append(noise_V)
        return ground, clean, noised, maps

    def transform_reverse(self, view_idx:int=0):
        return self.views[view_idx].matmul(self.maps[view_idx])
    
    def __getitem__(self, index):
        return (self.views[index], torch.tensor([0]), torch.tensor([1.]))
        
    def __len__(self):
        return self.length