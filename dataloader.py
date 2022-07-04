import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import ImageDataset
from monai.transforms import (
    Spacing,
    ResizeWithPadOrCrop,
    NormalizeIntensity,
    AddChannel,
    RandScaleIntensity,
    RandShiftIntensity,
    ToTensor,
    Compose,
    Orientation,
    Resize,
    ScaleIntensity 
)


import pytorch_lightning as pl


class TestCompose(Compose):
    def __call__(self, data, meta):
        data = self.transforms[0](data)
        data, _, meta["affine"] = self.transforms[1](data, meta["affine"])# spacing
        data, _, meta["affine"] = self.transforms[2](data, meta["affine"])# Orientation
        data = self.transforms[3](data)  # reisze
        data = self.transforms[4](data) # NormalizeIntensity

        if len(self.transforms) > 6: 
            data = self.transforms[5](data) # RandScaleIntensity
            data = self.transforms[6](data) # RandShiftIntensity

        data = self.transforms[-1](data) # totensro

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

        train_transform = TestCompose(
        [
            #ScaleIntensity(),
            AddChannel(),
            Spacing(
                pixdim=(1.5,1.5,6),
            ),
            Orientation(axcodes="RAS"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            #Resize((256,256,36)),
            ResizeWithPadOrCrop((209, 220,  47)),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            RandScaleIntensity(factors=0.1, prob=0.5),
            RandShiftIntensity(offsets=0.1, prob=0.5),
            ToTensor(),
        ])
        val_transform = TestCompose(
        [
            #ScaleIntensity(),
            AddChannel(),
            Spacing(
                pixdim=(1.5,1.5,6),
            ),
            Orientation(axcodes="RAS"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            #Resize((256,256,36)),
            ResizeWithPadOrCrop((209, 220,  47)),
            ToTensor(),
        ])

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
