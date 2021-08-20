
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
    def __init__(self, load_weights=False, verbose=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        self.backend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=3)
        self.backend_depth = make_layers(self.backend_feat, in_channels=3)
        self.frontend = make_layers(self.frontend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.backend.state_dict().items())):
                list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        
        
        bilinear = False
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear, dilation=True)
        self.up2 = Up(256, 128 // factor, bilinear, dilation=True)
        
        self.up1_1 = Up(512, 256 // factor, bilinear, dilation=True)
        self.dconv2_1 = OutConv(256, 128)
        
        self.dconv1 = OutConv(1024, 256)
        self.dconv2 = OutConv(256, 64)
        self.dconv3 = OutConv(64, 1)

        self.out1 = OutConv(128, 1)
        self.out1_1 = OutConv(128, 1)

        self.info(verbose=verbose)


    def forward(self, x_im, x_depth, training=True):
        device = x_im.device
        x = self.backend(x_im)

        x0 = self.frontend(x)
        x0 = self.output_layer(x0)
        #x0 = nn.ReLU()(x0)
        x0 = x0.squeeze(1)  # count_0

        x0_flat = x0.sum(dim=[1, 2]).view(x0.shape[0], 1)

        x1 = self.up1_1(x)
        x1 = self.dconv2_1(x1)
        x1 = self.out1_1(x1)
        #x1 = nn.ReLU()(x1)
        x1 = x1.squeeze(1) # count_1

        x1_flat = x1.sum(dim=[1, 2]).view(x1.shape[0], 1)

        x2 = self.up1(x)
        x2 = self.up2(x2) 
        x2 = self.out1(x2) 
        #x2 = nn.ReLU()(x2)
        x2 = x2.squeeze(1) # localization

        x2_flat = x2.sum(dim=[1, 2]).view(x2.shape[0], 1)

        x_flat_tot = torch.cat((x0_flat, x1_flat, x2_flat), dim=1)
        
        x_depth = self.backend_depth(x_depth)
        x_depth = torch.cat([x, x_depth], dim=1)
        #x_depth = x + x_depth
        x_depth = self.dconv1(x_depth) ############
        #x_depth = self.dconv1(x)  ############
        x_depth = nn.ReLU()(x_depth)
        x_depth = self.dconv2(x_depth)
        x_depth = nn.ReLU()(x_depth)
        x_depth = self.dconv3(x_depth)
        
        x_att = nn.Flatten()(x_depth)
        x_att = nn.Linear(x_att.shape[-1], 32, device=device)(x_att)
        x_att = nn.ReLU()(x_att)
        x_att = nn.Linear(x_att.shape[-1], 3, device=device)(x_att)
        x_att = nn.Softmax()(x_att)

        x_flat_tot = x_flat_tot * x_att
        #x_flat_tot = torch.sum(x_flat_tot, 1)
        x_flat_tot = torch.mean(x_flat_tot, 1)
            
        return x0, x1, x2, x_flat_tot


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

    def __init__(self, in_channels, out_channels, bilinear=True, dilation=False):
        super().__init__()

        if dilation:
            d_rate = 2
        else:
            d_rate = 1

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, d_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels, d_rate)


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

    def __init__(self, in_channels, out_channels, mid_channels=None, d_rate=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=d_rate, dilation=d_rate),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=d_rate, dilation=d_rate),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=d_rate, dilation=d_rate),
            # nn.BatchNorm2d(out_channels),
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
    def __init__(self, model):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        #self.MSELoss = nn.MSELoss(reduction='mean') 
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))


    def __call__(self, p0, p1, p2, total_count, targets): 
        device = targets.device
        
        # Losses
        lcount_0 = torch.zeros(1, device=device)
        lcount_1 = torch.zeros(1, device=device)
        lcount_2 = torch.zeros(1, device=device)
        lcount = torch.zeros(1, device=device)
        
        tcount_0 = torch.zeros_like(p0, device=device)  
        tcount_1 = torch.zeros_like(p1, device=device)  
        tcount_2 = torch.zeros_like(p2, device=device)  
        #tcount = torch.zeros((p0.shape[0], p0.shape[1]*8, p0.shape[2]*8), device=device)   

        
        indices_0, indices_1, indices_2 = self.build_targets(targets, p2.size(1)*2, p2.size(2)*2) 
        
        gj, gi = indices_2[0]  
        
        b = 0
        n = gi.shape[0]  
        if n:
            tcount_2[b, gi, gj] = 1 

            indices_0 = indices_0[0]
            indices_0 = torch.stack(indices_0)
            nonempty, count = torch.unique(indices_0, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_0[b, indx[1], indx[0]] = float(count[k])
            
            indices_1 = indices_1[0]
            indices_1 = torch.stack(indices_1)
            nonempty, count = torch.unique(indices_1, dim=1, return_counts=True) 

            for k in range(nonempty.size(1)):
                indx = nonempty[:, k]
                tcount_1[b, indx[1], indx[0]] = float(count[k])
            

        H0 = 1/4 
        H1 = 1/9 
        H2 = 1/121 
        H3 = 1/(10000)  

        lcount_0 += (self.MSELoss(p0, tcount_0) * H0) 
        lcount_1 += (self.MSELoss(p1, tcount_1) * H1) 
        lcount_2 += (self.MSELoss(p2, tcount_2) * H2)

        tcount = tcount_0.sum()
        lcount += (self.MSELoss(total_count, tcount) * H3)
        
        return  (lcount + lcount_0 + lcount_1 + lcount_2), lcount_0.detach(), lcount_1.detach(), lcount_2.detach()


    def build_targets(self, targets, s0, s1):
        # Build targets for compute_loss()
        device = targets.device
        
        gain = torch.ones(2, device=device)  # normalized to gridspace gain
        
        ##### count
        indices_0 = []
        gain = torch.tensor((s0//8, s1/8), device=device) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_0.append((gj, gi))  

        ##### count
        indices_1 = []
        gain = torch.tensor((s0//4, s1/4), device=device) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_1.append((gj, gi))  
        
        ##### loc
        indices_2 = []
        gain = torch.tensor((s0//2, s1/2), device=device) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        gij = t.long()
        gi, gj = gij.T  # grid xy indices

        indices_2.append((gj, gi))  

        return indices_0, indices_1, indices_2

