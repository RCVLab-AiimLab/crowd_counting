
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
        self.regress = nn.Sequential(nn.Linear(self.in_size, self.in_size//2), 
                                    nn.LeakyReLU(),
                                    nn.Linear(self.in_size//2, self.in_size//2), 
                                    nn.LeakyReLU(),
                                    nn.Linear(self.in_size//2, 1))
        
        self.info(verbose=verbose)


    def forward(self, x, training=True):
        x = self.backend(x)
        x = self.frontend(x)
        x = self.output_layer(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x0 = x[..., 0]
        x1 = x[..., 1]
        #x = x.squeeze(1)
        x1 = x1.view(-1, self.in_size)
        x1 = self.regress(x1)

        return x0, x1


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
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.in_size = in_size

    def __call__(self, p0, p1, targets): 
        device = targets.device
        lloc = torch.zeros(1, device=device)
        lcell = torch.zeros(1, device=device)
        indices = self.build_targets(targets) 
        
        # Losses
        b, gj, gi = indices[0]  
        tloc = torch.zeros_like(p0, device=device)  
        tcell = torch.zeros_like(p1, device=device)   

        n = b.shape[0]  
        if n:
            #tobj[b, gi, gj] = 1 

            nonempty, count = torch.unique(b, return_counts=True) 

            for k, bi in enumerate(nonempty):
                indx = torch.where(b==bi)
                tloc[bi, gi[indx], gj[indx]] += 1
                tcell[bi, 0] = float(count[k])

        lloc += self.MSELoss(p0, tloc)
        lcell += self.MSELoss(p1, tcell)
        
        #bs = tobj.shape[0]
        
        return  lloc + lcell, lloc.detach(), lcell.detach()


    def build_targets(self, targets):
        # Build targets for compute_loss()
        indices = []

        gain = torch.ones(5, device=targets.device)  # normalized to gridspace gain
        
        gain[1:3] = torch.tensor((self.in_size//8, self.in_size//8)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  
        gxy = t[:, 1:3]  # grid xy


        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices.append((b, gj, gi))  

        return indices

