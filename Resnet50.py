import torchvision.models as models
import torch
import torch.nn as nn

class resnet50(nn.Module):
    def __init__(self,input_chanel=1,pretrained=True,num_class=1):
        super().__init__()
        self.model_ft = models.video.r3d_18(pretrained=pretrained)
        self.model_ft.stem[0] = nn.Conv3d(input_chanel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model_ft.fc = nn.Linear(in_features=512, out_features=num_class, bias=True)
    
    def forward(self,x):
        out = self.model_ft(x)
        return out 
