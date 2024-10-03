#-*- coding:utf-8 -*-

from sklearn.preprocessing import MinMaxScaler
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

class NiftiImageDataset(Dataset):
    def __init__(self, 
                imagefolder_path:str, 
                input_size:int, 
                depth_size:int, 
                transform=None
        ):
        self.imagefolder_path = imagefolder_path
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder_path, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path:str):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # image in [0, 1] normalize
        return img
    
    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img


    def plot_samples(self, n_slice:int=15, n_row:int=4):
        if self.transform is not None:
            self.transform = None
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice], cmap='gray')
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile) # image in [-1, 1] : this process is required to resize the image.
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img
    
class NiftiImagePairedDataset(Dataset):
    """
        dataset folder example:
            - ct: 
                - 0.nii.gz,
                - 1.nii.gz,
                - 2.nii.gz, ...
            - mri:
                - 0.nii.gz,
                - 1.nii.gz,
                - 2.nii.gz, ...
    """
    def __init__(self, 
                input_folder_path:str, 
                target_folder_path:str,
                input_size:int, 
                depth_size:int, 
                transform=None
        ):
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_files = np.sort(glob(os.path.join(input_folder_path, '*.nii.gz')))
        self.target_files = np.sort(glob(os.path.join(target_folder_path, '*.nii.gz')))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path:str):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # image in [0, 1] normalize
        return img
    
    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img


    def plot_samples(self, n_slice:int=15, n_row:int=4):
        if self.transform is not None:
            self.transform = None
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice], cmap='gray')
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        input_file = self.input_files[index]
        target_file = self.target_files[index]
        img = self.read_image(input_file)
        target = self.read_image(target_file)
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(input_file) # image in [-1, 1] : this process is required to resize the image.
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

            target = tio.ScalarImage(target_file)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            target = np.asarray(cop(target))[0]

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        return img, target

if __name__ == '__main__':
    transform = T.Compose([
        T.Lambda(lambda t: torch.tensor(t).float()),
        T.Lambda(lambda t: t.unsqueeze(0)), # add channel
        T.Lambda(lambda t: (t * 2) - 1), # img in [-1, 1] normalizing any case -> max min ranges depend on the cases of the sizes?
        T.Lambda(lambda t: t.permute(0, 3, 1, 2)), # HWD -> DHW
        T.Lambda(lambda t: t.unsqueeze(0)) # add batch dim
    ])
    dataset = NiftiImagePairedDataset(
        "./data/ct", 
        "./data/mri", 
        input_size=128, 
        depth_size=128, 
        transform=transform
    )
    x, y = dataset[0]
    print("input>", x.shape, torch.min(x), torch.max(x))
    print("target>", y.shape, torch.min(y), torch.max(y))
    # dataset.plot_samples()