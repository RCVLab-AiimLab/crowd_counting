
import numpy as np 
import math
from copy import deepcopy
import yaml 
import torch
import torch.nn as nn 
import cv2 
from itertools import product, starmap
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False, in_size=128, verbose=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        self.backend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=3)
        self.frontend = make_layers(self.frontend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)
        
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.backend.state_dict().items())):
                list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        
        self.in_size = (in_size // 8)**2
        
        bilinear = False
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        
        self.out1 = OutConv(128, 1)

        self.info(verbose=verbose)


    def forward(self, x_im, training=True):
        
        x = self.backend(x_im)

        x01 = self.frontend(x)
        x01 = self.output_layer(x01) 

        x0 = x01[:, 0, :, :]
        x0 = x0.view(x0.size(0), -1).sum(1, keepdim=True)  # count 0

        x1 = x01[:, 1, :, :] # count 1

        x2 = self.up1(x)
        x2 = self.up2(x2) 
        x2 = self.out1(x2) 
        x2 = x2.squeeze(1) # localization
        
        if not training: 
            x2 = x2.sigmoid()
            
        return x0, x1, x2


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)


    def forward(self, x1):
        x1 = self.up(x1)

        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  


def model_info(model, verbose=False, img_size=128):
    # Model information. 
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    
    fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, in_size=128):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        #self.MSELoss = nn.MSELoss(reduction='mean') 
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.in_size = in_size

        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def __call__(self, p0, p1, p2, targets): 
        device = targets.device
        
        # Losses
        lcount_0 = torch.zeros(1, device=device)
        lcount_1 = torch.zeros(1, device=device)
        lloc = torch.zeros(1, device=device)
        
        tcount_0 = torch.zeros_like(p0, device=device)  
        tcount_1 = torch.zeros_like(p1, device=device)  
        tloc = torch.zeros_like(p2, device=device)   
        
        indices_count, indices_loc = self.build_targets(targets) 
        
        b, gj, gi = indices_loc[0]  
        
        n = b.shape[0]  
        if n:
            tloc[b, gi, gj] = 1 

            indices_count = indices_count[0]
            indices_count = torch.stack(indices_count)
            nonempty, count = torch.unique(indices_count[0, :], dim=0, return_counts=True) 

            for k in range(nonempty.size(0)):
                indx = nonempty[k]
                tcount_0[indx, :] = float(count[k])
            
            nonempty, count = torch.unique(indices_count, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_1[indx[0], indx[2], indx[1]] = float(count[k])
            
        bs = n * self.in_size * 2

        H0 = 1/self.in_size
        H1 = 1/16
        H2 = self.in_size

        lcount_0 += (self.MSELoss(p0, tcount_0) * H0) 
        lcount_1 += (self.MSELoss(p1, tcount_1) * H1) 
        lloc += (self.BCEobj(p2, tloc) * H2)
        
        return  (lcount_0 + lcount_1 + lloc), lcount_0.detach(), lcount_1.detach(), lloc.detach()


    def build_targets(self, targets):
        # Build targets for compute_loss()
        
        gain = torch.ones(5, device=targets.device)  # normalized to gridspace gain
        
        ##### count
        indices_0 = []
        gain[1:3] = torch.tensor((self.in_size//8, self.in_size//8)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  
        gxy = t[:, 1:3]  # grid xy

        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices_0.append((b, gj, gi))  
        
        ##### loc
        indices_1 = []
        gain[1:3] = torch.tensor((self.in_size//2, self.in_size//2)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  
        gxy = t[:, 1:3]  # grid xy

        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices_1.append((b, gj, gi))  

        return indices_0, indices_1

