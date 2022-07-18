import torchvision.models as models
import torch
import torch.nn as nn
import monai
import numpy as np
import random
import pytorch_lightning as pl
import renset_med

class resnet50(nn.Module):
    def __init__(self,input_chanel=1,pretrained=True,num_class=1):
        super().__init__()
        self.model_ft = models.video.r3d_18(pretrained=pretrained)
        prev_w = self.model_ft.stem[0].weight
        self.model_ft.stem[0] = nn.Conv3d(input_chanel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model_ft.stem[0].weight.data = prev_w.data.sum(dim=1 ,keepdim=True)
        self.model_ft.fc = nn.Linear(in_features=512, out_features=num_class, bias=True)
    
    def forward(self,x):
        out = self.model_ft(x)
        return out 

class resnet50_m(nn.Module):
    def __init__(self,sample_input_D,sample_input_H,sample_input_W,num_classes,no_cuda,shortcut_type,pretrained_path=None):
        super().__init__()
        self.model_ft = renset_med.resnet50(
            sample_input_D = sample_input_D, 
            sample_input_H = sample_input_H,
            sample_input_W = sample_input_W,
            num_classes = num_classes,
            no_cuda = no_cuda,
            shortcut_type = shortcut_type
        )
        if pretrained_path is not None:
            pretrain = torch.load(pretrained_path)
            model_dict = self.model_ft.state_dict()
            pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model_ft.load_state_dict(model_dict)
    
    def forward(self,x):
        out = self.model_ft(x)
        return out 
    
class resnet10_m(nn.Module):
    def __init__(self,sample_input_D,sample_input_H,sample_input_W,num_classes,no_cuda,shortcut_type,pretrained_path=None):
        super().__init__()
        self.model_ft = renset_med.resnet10(
            sample_input_D = sample_input_D, 
            sample_input_H = sample_input_H,
            sample_input_W = sample_input_W,
            num_classes = num_classes,
            no_cuda = no_cuda,
            shortcut_type = shortcut_type
        )
        if pretrained_path is not None:
            pretrain = torch.load(pretrained_path)
            model_dict = self.model_ft.state_dict()
            pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model_ft.load_state_dict(model_dict)
    
    def forward(self,x):
        out = self.model_ft(x)
        return out 
