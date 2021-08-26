
import numpy as np 
import math
from copy import deepcopy
import yaml 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import cv2 
from itertools import product, starmap
from torchvision import models
from spp import spatial_pyramid_pool
# import segmentation_models_pytorch as smp 


class CSRNet(nn.Module):
    def __init__(self, load_weights=False, in_size=128, verbose=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        self.backend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.decode = ['U', 256, 256, 'U', 128, 128, 'U', 64, 64]
        self.frontend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=3)
        self.backend_split = make_layers_split(self.backend_feat, in_channels=3)
        # print(self.backend_split)
        self.decode_make = make_layers(self.decode, in_channels=512)
        self.frontend = make_layers(self.frontend_feat, in_channels=512, dilation=True)

        # self.de_back = make_layers(self.backend1)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.output_layer2 = nn.Conv2d(67, 64, kernel_size=3, padding=1, dilation=1)
        self.output_layer3 = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.backend.state_dict().items())):
                list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                list(self.backend_split.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                list(self.backend_split.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        print(torch.allclose(mod.state_dict()['features.0.weight'], self.backend_split.state_dict()['0.0.weight']))   

        self.avg = nn.AdaptiveAvgPool2d((1, 512))

        self.output_num = [10]
        
        self.info(verbose=verbose)


    def forward(self, x, x_depth, training=True):
        l=[]
        img = torch.clone(x)
        for layer in self.backend_split:
            img = layer(img)
            l.append(img)
        # print(len(l))
        xx1, xx2, xx3, xx4 = l

        device = x.device
        d_rate = 1

        up_xx0 = nn.Upsample(scale_factor=2, mode='bilinear')(xx4)

        up_xx0_cat = torch.cat((xx3, up_xx0), dim=1)
        
        dec_xx0 = nn.Conv2d(768, 256, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(up_xx0_cat)
        dec_xx0 = nn.BatchNorm2d(256, device=device)(dec_xx0)
        dec_xx0 = F.relu(dec_xx0)
        dec_xx0 = nn.Conv2d(256, 256, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(dec_xx0)
        dec_xx0 = nn.BatchNorm2d(256, device=device)(dec_xx0)
        dec_xx0 = F.relu(dec_xx0)
        dec_xx0 = nn.Conv2d(256, 256, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(dec_xx0)
        dec_xx0 = nn.BatchNorm2d(256, device=device)(dec_xx0)
        dec_xx0 = nn.ReLU()(dec_xx0)

        up_xx1 = nn.Upsample(scale_factor=2, mode='bilinear')(dec_xx0)

        up_xx1_cat = torch.cat((xx2, up_xx1), dim=1)

        dec_xx1 = nn.Conv2d(384, 128, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(up_xx1_cat)
        dec_xx1 = nn.BatchNorm2d(128, device=device)(dec_xx1)
        dec_xx1 = F.relu(dec_xx1)
        dec_xx1 = nn.Conv2d(128, 128, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(dec_xx1)
        dec_xx1 = nn.BatchNorm2d(128, device=device)(dec_xx1)
        dec_xx1 = nn.ReLU()(dec_xx1)

        up_xx2 = nn.Upsample(scale_factor=2, mode='bilinear')(dec_xx1)

        up_xx2_cat = torch.cat((xx1, up_xx2), dim=1)

        dec_xx2 = nn.Conv2d(192, 64, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(up_xx2_cat)
        dec_xx2 = nn.BatchNorm2d(64, device=device)(dec_xx2)
        dec_xx2 = F.relu(dec_xx2)
        dec_xx2 = nn.Conv2d(64, 64, kernel_size=3, padding=d_rate, dilation=d_rate, device=device)(dec_xx2)
        dec_xx2 = nn.BatchNorm2d(64, device=device)(dec_xx2)
        decoded_x = nn.ReLU()(dec_xx2)
        
        # print(x.shape)
        # print(xx4.shape)
        # decoded_x = self.decode_make(xx)
        x0 = self.output_layer(decoded_x) 
        # x0 = nn.ReLU()(x0)
        xx4 = self.backend(x)
        x_density = self.frontend(xx4)
        x_density = self.output_layer(x_density) 
        # x_density = nn.Conv2d(64, 1, kernel_size=1, device = device)(x_density)
        # x_density = nn.ReLU()(x_density)


        # x0_x = torch.cat((x0, x), dim = 1)
        # x0_x = self.output_layer2(x0_x) 
        # x0_x = nn.ReLU()(x0_x)
        # x0_x = self.output_layer3(x0_x) 
        # x0 = nn.ReLU()(x0_x)

        density_flat = x_density.sum(dim = [2, 3]).view(x0.shape[0],1)
        x0_flat = x0.sum(dim = [2, 3]).view(x0.shape[0],1)
        x_flat_tot = torch.cat((x0_flat, density_flat), dim = 1)

        x_flat_tot = torch.mean(x_flat_tot, dim = 1)


        '''# uncomment if binary loss is used
        if not training: 
            x0 = x0.sigmoid()
            x1 = x1.sigmoid()
            x2 = x2.sigmoid()
        '''    
        return x0, x_flat_tot, x_density


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





def make_layers_split(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    layer = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.Sequential(*layer))
            layer = []
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layer += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if len(layer)!=0:  # add any leftover layers
        layers.append(nn.Sequential(*layer))   
    # print(len(layer))
    # print(len(layers))
    return nn.ModuleList(layers)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
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
        self.HUBERLoss = nn.HuberLoss(reduction='sum', delta=1.35)
        self.in_size = in_size

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def __call__(self, p0, targets, count, density, density_out): 
        device = targets.device
        
        lcount_0 = torch.zeros(1, device=device)
        lcount = torch.zeros(1, device=device)
        lcount_density = torch.zeros(1, device=device)

        targets_sum = targets.sum()
        lcount_0 += self.BCEobj(p0.float(), targets.float())
        lcount_density += self.MSELoss(density.float(), density_out.float())
        lcount += self.MSELoss(count.float(), targets_sum.float())
        # print('*'*1000)
        # print(density_out.shape)
        # print(density.shape)
        # print('*'*1000)
        # lcount_density += self.MSELoss(density_out.squeeze(1), density)
        # print('*'*1000)
        # print(count.shape)
        # print(tcount.sum(dim = [1, 2]).view(tcount.shape[0], 1).shape)
        # lcount += self.MSELoss(count.view(count.shape[0], 1), tcount.sum(dim = [1, 2]).view(tcount.shape[0], 1))
        # print('tcount', tcount.sum())
        # print('count', count.sum())
        # print('tcount_0', tcount_0.sum())
        # print('tcount_1', tcount_1.sum())
        # print('tcount_2', tcount_2.sum())
        lcount_1 = torch.zeros(1, device=device)
        lcount_2 = torch.zeros(1, device=device)
    
        return lcount_density

'''
x = self.backend(x)
        xx = self.frontend(x)

        x0 = self.output_layer(xx) 
        # x0 = nn.ReLU()(x0)
        x0 = x0.squeeze(1) # count_0
        x0_flat = x0.sum(dim = [1, 2]).view(x0.shape[0],1)
        print('x0 shape', x0.shape)

        x = self.up1(x)
        x1 = self.outc1(x)
        # x1 = nn.LeakyReLU()(x1)
        # x1 = self.outc3(x1)
        # x1 = nn.ReLU()(x1)
        x1 = x1.squeeze(1) # count_1
        x1_flat = x1.sum(dim = [1, 2]).view(x1.shape[0],1)

        x2 = self.up2(x) 
        # x2 = self.outc2(xx)
        # x2 = nn.LeakyReLU()(x2)
        # # x2 = self.outc3(x2)
        # # x2 = nn.ReLU()(x2)
        # x2 = x2.squeeze(1) # count_2
        # x2_flat = x2.sum(dim = [1, 2]).view(x2.shape[0],1)'''