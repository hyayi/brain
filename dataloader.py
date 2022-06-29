import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import ImageDataset
from monai.transforms import AddChannel,  ScaleIntensity, EnsureType, Spacing,Compose,RandAdjustContrast

import torchio as tio
from torchio.transforms import CropOrPad, ZNormalization

import pytorch_lightning as pl

class TestCompose(Compose):
    def __call__(self, data, meta):
        data = self.transforms[0](data) # addchannell
        data, _, meta["affine"] = self.transforms[1](data, meta["affine"])# spacing
        data = self.transforms[2](data)  # pad
        if len(self.transforms) == 6: 
            data = self.transforms[3](data) # randconst
            data = self.transforms[4](data) # scale
            data = self.transforms[5](data) # totensro
        else :
            data = self.transforms[3](data) # scale
            data = self.transforms[4](data) # totensro

        return data,meta


class BrainDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./", batch_size=16, num_workers=3,pin_memory=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def prepare_data(self):

        self.train = pd.read_csv(f"{self.data_dir}/train.csv")
        self.val = pd.read_csv(f"{self.data_dir}/validation.csv") 
        self.test = pd.read_csv(f"{self.data_dir}/test.csv")
        
        self.train['image'] = self.train['image'].apply(lambda x : f"{x}.nii.gz")
        self.val['image'] = self.val['image'].apply(lambda x : f"{x}.nii.gz")
        self.test['image'] = self.test['image'].apply(lambda x : f"{x}.nii.gz")
       
        self.train['image'] = self.train['image'].apply(lambda x : os.sep.join([self.data_dir, x]))
        self.val['image'] = self.val['image'].apply(lambda x : os.sep.join([self.data_dir, x]))
        self.test['image'] = self.test['image'].apply(lambda x : os.sep.join([self.data_dir, x]))

    def setup(self, stage = None):

        train_transform = TestCompose([AddChannel() , Spacing(pixdim=(1, 1, 2)),CropOrPad((256, 256, 35)),RandAdjustContrast(),ZNormalization(masking_method=tio.ZNormalization.mean),EnsureType()])
        val_transform =  TestCompose([AddChannel(), Spacing(pixdim=(1, 1, 2)),CropOrPad((256, 256, 35)),ZNormalization(masking_method=tio.ZNormalization.mean),EnsureType()])

        if stage == 'fit' or stage is None:
            self.train_ds = ImageDataset(image_files=self.train['image'], labels=np.expand_dims(self.train['label'].values, axis=1).astype(np.float32), transform=train_transform,image_only=False,transform_with_metadata=True)
            self.validation_ds = ImageDataset(image_files=self.val['image'], labels=np.expand_dims(self.val['label'].values, axis=1).astype(np.float32), transform=val_transform,image_only=False,transform_with_metadata=True)
        if stage == 'test' or stage is None:
            self.test_ds = ImageDataset(image_files=self.test['image'], labels=np.expand_dims(self.test['label'].values, axis=1).astype(np.float32), transform=val_transform,image_only=False,transform_with_metadata=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.validation_ds,batch_size=self.batch_size,num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,num_workers=self.num_workers, pin_memory=self.pin_memory)
