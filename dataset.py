#-*- coding:utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
from torchvision.datasets import MovingMNIST
from torch.utils.data import Dataset
from torchvision import transforms as T
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os

class MovingMNIST(Dataset):
    def __init__(self, data_file:str, num_frames:int=10, transform=None):
        super().__init__()
        self.data_file = data_file
        self.datas = np.load(data_file)
        # (t, N, H, W) -> (N, t, C, H, W)
        self.datas = self.datas.transpose(1, 0, 2, 3)[:, :, None, ...]
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        y = self.datas[index, :self.num_frames, ...].astype(np.int32) # = x_T
        x = self.datas[index, self.num_frames:, ...].astype(np.int32) # = x_0
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
    
if __name__ == '__main__':
    transform = T.Compose([
        T.Lambda(lambda t: torch.tensor(t).float()),
        T.Lambda(lambda t: (t / 255. * 2) - 1), # img in [-1, 1] normalizing
        T.Lambda(lambda t: t.permute(1, 0, 2, 3)), # TCHW -> CTHW
        T.Lambda(lambda t: t.unsqueeze(0)) # add batch dim
    ])
    dataset = MovingMNIST("./data/mnist_test_seq.npy", transform=transform)
    print("len>", len(dataset))
    x, y = dataset[0]
    print(x.shape, y.shape)
    print(">", torch.min(x), torch.max(y))

    from torchvision.io import write_video
    video = 255*0.5*(x.permute(0, 2, 1, 3, 4) + 1)[0]
    video = video.repeat(1, 3, 1, 1)
    video = video.permute(0, 2, 3, 1)
    print(">", video.shape)
    write_video("data/test.mp4", video, fps=5)