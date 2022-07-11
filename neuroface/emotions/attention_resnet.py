import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torchvision import models

import os
import gdown


class AttentionResnet(nn.Module):
    
    def __init__(self, num_class=8, num_head=4, pretrained=None, device=None):
        super(AttentionResnet, self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        
        for idx in range(num_head):
            setattr(self,'cat_head{}'.format(idx), CrossAttentionHead())
        
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        
        if pretrained:
            path = 'https://drive.google.com/uc?export=view&id=17lzsrHyuSGd2cZuNHdAAPCw6JsrjgFIn'
            model_name = 'affecnet8_epoch5_acc0.6209.pth'
            model_dir = os.path.join(get_torch_home(), 'checkpoints')
            os.makedirs(model_dir, exist_ok=True)
            
            cached_file = os.path.join(model_dir, os.path.basename(model_name))
            
            if not os.path.exists(cached_file):
                gdown.download(path, cached_file, quiet=False)
            
            state_dict = torch.load(cached_file)
            self.load_state_dict(state_dict['model_state_dict'], strict=True)
        
        self.device = torch.device('cpu')
        
        if device is not None:
            self.device = device
            self.to(device)
    
    def forward(self, x):
        x = self.features(x)
        heads = []
        
        for i in range(self.num_head):
            heads.append(getattr(self, 'cat_head{}'.format(idx))(x))
        
        heads = torch.stack(heads).permute([1, 0, 2])
        
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)
        
        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
        
        return out, x, heads


class CrossAttentionHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)
        
        return ca


class SpatialAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y
        
        return out


class ChannelAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )
    
    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y
        
        return out


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    
    return torch_home