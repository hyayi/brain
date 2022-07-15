import torch
import pandas as pd
import numpy as np

class MRSDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transforms):
        self.data_df = data_df
        self.transforms = transforms

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):

        image_dict = {'img' : self.data_df['image'][index]}
        label = torch.as_tensor(self.data_df['label'][index],dtype=torch.float)
        return self.transforms(image_dict), label
