import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import ImageDataset
from monai.transforms import LoadImaged, AddChanneld, Orientationd, ScaleIntensityd, RandRotated,Resized, RandShiftIntensityd, EnsureTyped
import numpy as np
import random
import pytorch_lightning as pl
from dataset import MRSDataset


class BrainDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./", batch_size=16, num_workers=3,pin_memory=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


    def prepare_data(self):

        self.train = pd.read_csv(f"{self.data_dir}/train.csv")
        self.val = pd.read_csv(f"{self.data_dir}/val.csv") 
        self.test = pd.read_csv(f"{self.data_dir}/test.csv")
        
        self.train['image'] = self.train['image'].apply(lambda x : f"{x}.nii.gz")
        self.val['image'] = self.val['image'].apply(lambda x : f"{x}.nii.gz")
        self.test['image'] = self.test['image'].apply(lambda x : f"{x}.nii.gz")
       
        self.train['image'] = self.train['image'].apply(lambda x : os.sep.join([self.data_dir, x]))
        self.val['image'] = self.val['image'].apply(lambda x : os.sep.join([self.data_dir, x]))
        self.test['image'] = self.test['image'].apply(lambda x : os.sep.join([self.data_dir, x]))



    def setup(self, stage = None):

        train_transforms = Compose(
            [
                LoadImaged(keys="img"),
                AddChanneld(keys="img"),
                Orientationd(axcodes="SPL"),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(128, 256, 256)),
                RandRotated(keys=["img"], range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandShiftIntensityd(keys=["img"],offsets=0.1, prob=0.5),
                EnsureTyped(keys=["img"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys="img"),
                AddChanneld(keys="img"),
                Orientationd(axcodes="SPL"),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(128, 256, 256)),
                EnsureTyped(keys=["img"]),
            ]
        )

        if stage == 'fit' or stage is None:
            self.train_ds = MRSDataset(data_df=self.train[['image','label']], transforms= train_transform)
            self.validation_ds = MRSDataset(data_df=self.val[['image','label']], transforms= val_transform)

        if stage == 'test' or stage is None:
            self.test_ds = MRSDataset(data_df=self.test[['image','label']], transforms= val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.validation_ds,batch_size=self.batch_size,num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,num_workers=self.num_workers, pin_memory=self.pin_memory)
