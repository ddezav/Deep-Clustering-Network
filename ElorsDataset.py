import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class ElorsDataSet(Dataset):
    def __init__(dir):
        xy = np.loadtxt(dir,delimiter=",",dtype=np.float32)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
    def __len__(self):
        return self.n_samples