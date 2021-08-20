
import numpy as np 
import math
from copy import deepcopy
import yaml 
import torch
import torch.nn as nn 
import cv2 
from itertools import product, starmap
from torchvision import models
import segmentation_models_pytorch as smp 


class CSRNet(nn.Module):
    def __init__(self, load_weights=False, in_size=128, verbose=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        self.backend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=3)
        self.frontend = make_layers(self.frontend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.backend.state_dict().items())):
                list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        
        self.in_size = (in_size // 8)**2
        
        bilinear = False
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 128 // factor, bilinear)
        self.up2 = Up(512, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32 // factor, bilinear)
        self.out1 = OutConv(128, 1)
        self.out2 = OutConv(32, 3)
        
        self.info(verbose=verbose)


    def forward(self, x_im, training=True):
        x = self.backend(x_im)

        x0 = self.frontend(x)
        x0 = self.output_layer(x0) 
        x0 = x0.squeeze(1) # count

        x1 = self.up1(x)
        #x1 = self.up2(x1) 
        x1 = self.out1(x1) 
        x1 = x1.squeeze(1) # localization

        #x2 = self.up2(x)
        #x2 = self.up3(x2)
        #x2 = self.up4(x2)
        #x2 = self.out2(x2) 

        if not training: 
            #x0 = x0.sigmoid()
            #x1 = x1.sigmoid()
            pass
            
        return x0, x1, [x_im, x_im]


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


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, in_size=128):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        #self.MSELoss = nn.MSELoss(reduction='mean') 
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.MAELoss = nn.L1Loss(reduction='sum')
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.in_size = in_size

    def __call__(self, p0, p1, p2, targets): 
        device = targets.device
        
        lcount = torch.zeros(1, device=device)
        lloc = torch.zeros(1, device=device)
        lrecon = torch.zeros(1, device=device)
        
        indices_loc, indices_count = self.build_targets(targets) 
        
        tcount = torch.zeros_like(p0, device=device)  
        tloc = torch.zeros_like(p1, device=device)   
        
        # Losses
        b, gj, gi = indices_loc[0]  

        n = b.shape[0]  
        if n:
            #tloc[b, gi, gj] = 1 
            
            indices_count = indices_count[0]
            indices_count = torch.stack(indices_count)
            nonempty, count = torch.unique(indices_count, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount[indx[0], indx[2], indx[1]] = float(count[k])

            indices_loc= indices_loc[0]
            indices_loc = torch.stack(indices_loc)
            nonempty, count = torch.unique(indices_loc, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tloc[indx[0], indx[2], indx[1]] = float(count[k])
            

        lcount += self.MAELoss(p0.sum(), tcount.sum())
        lloc += self.MSELoss(p1, tloc)
        #lrecon += self.MSELoss(p2[0], p2[1])
        #lcell += self.BCEobj(p1, tcell)
        
        bs = tloc.shape[0] * self.in_size
        
        return  (lcount + lloc), lcount.detach(), lloc.detach(), lcount.detach()


    def build_targets(self, targets):
        # Build targets for compute_loss()
        indices_loc = []

        gain = torch.ones(5, device=targets.device)  # normalized to gridspace gain
        
        gain[1:3] = torch.tensor((self.in_size//4, self.in_size//4)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  
        gxy = t[:, 1:3]  # grid xy

        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices_loc.append((b, gj, gi))  
        
        #####
        indices_count = []
        gain[1:3] = torch.tensor((self.in_size//8, self.in_size//8)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  
        gxy = t[:, 1:3]  # grid xy

        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices_count.append((b, gj, gi))  
        #####

        return indices_loc, indices_count



