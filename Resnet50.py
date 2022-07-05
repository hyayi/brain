import torchvision.models as models
import torch
import torch.nn as nn
import monai
import numpy as np
import random
import pytorch_lightning as pl

pl.seed_everything(42)

def torch_seed(random_seed=42):

    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    
torch_seed(42)

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
    def __init__(self,n_input_channels=1,num_classes=1,pretrained_path=None):
        super().__init__()
        self.model_ft = monai.networks.nets.resnet50(n_input_channels=n_input_channels, num_classes=num_classes)
        if pretrained_path is not None:
            pretrain = torch.load(pretrained_path)
            pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
            self.model_ft.load_state_dict(pretrain['state_dict'], strict=False)
    
    def forward(self,x):
        out = self.model_ft(x)
        return out 
