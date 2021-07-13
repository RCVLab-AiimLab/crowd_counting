import numpy as np 
import math
from copy import deepcopy
import yaml 
import torch
import torch.nn as nn 
import cv2 


        
class Model(nn.Module):
    def __init__(self, model_file): 
        super(Model, self).__init__()

        with open(model_file) as f:
            self.model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        self.model, self.save = parse_model(deepcopy(self.model_dict))  

        initialize_weights(self)
        self.info()
    
    def forward(self, x, training=True):
        x = Shift_grid(x)
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if isinstance(m, Detect):
                x = m(x, not training)  # run
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        x = Max_probability(x)

        return x


    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def Shift_grid(x):
    pr = nn.functional.pad(x, (2,0,0,0), mode='constant', value=0)
    pd = nn.functional.pad(x, (0,0,2,0), mode='constant', value=0)
    pr_slice = pr[:,:,:,0:-2]
    pd_slice = pd[:,:,0:-2,:]
    x = torch.cat((x,pr_slice,pd_slice), dim=2)
    return x

def Max_probability(x):
    x_p,x_pr,x_pd = torch.chunk(x[:,0,:,:], chunks=3, dim=1)

    x_p_sr = nn.functional.pad(x_p, (1,0,0,0), mode='constant', value=0)
    x_p_sr = x_p_sr[:,:,0:-1]

    xr = torch.cat((x_p,x_pr,x_p_sr), dim=0)
    xr = torch.amax(xr,dim=0)
    xr = torch.unsqueeze(xr, dim=0)

    x_p_sd = nn.functional.pad(x_p, (0,0,1,0), mode='constant', value=0)
    x_p_sd = x_p_sd[:,0:-1,:]

    xd = torch.cat((x_p,x_pd,x_p_sd), dim=0)
    xd = torch.amax(xd,dim=0)
    xd = torch.unsqueeze(xd, dim=0)

    x = torch.cat((xr,xd), dim=0)
    x = torch.amax(x,dim=0)
    x = torch.unsqueeze(x, dim=0)
    return x

def model_info(model, verbose=False, img_size=640):
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


def parse_model(d):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))

    gd, gw = d['depth_multiple'], d['width_multiple']
    no = 1 
    ch = [3]
    layers, save, c2 = [], [], ch[-1]

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv]:
            c1, c2 = ch[f], args[0]

            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            #args[1] = [list(range(args[1] * 2))] * len(f)
            args = [args[2]]

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)            



class Detect(nn.Module):
    def __init__(self, ch=()):
        super(Detect, self).__init__()
        self.grid = [torch.zeros(1)]
        self.m = nn.ModuleList(nn.Conv2d(x, 2, 1) for x in ch)  # output conv


    def forward(self, x, inference):

        x = self.m[0](x[0])  # conv
        
        #bs, _, ny, nx = x.shape
        #x = x.view(bs, ny, nx)

        if inference:  # inference
            x[:, 0, :, :] = x[:, 0, :, :].sigmoid()#.squeeze()

        return x


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    if isinstance(x, list):
        x = x[0]
    return math.ceil(x / divisor) * divisor



class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.MSELoss = nn.MSELoss()
        

    def __call__(self, p, targets): 
        device = targets.device
        lobj = torch.zeros(1, device=device)
        lnum = torch.zeros(1, device=device)
        indices = self.build_targets(p, targets) 

        # Losses
        b, gj, gi = indices[0]  
        tobj = torch.zeros_like(p[:, 0, :, :], device=device)  
        tnum = torch.zeros_like(p[:, 0, :, :], device=device)  

        n = b.shape[0]  
        if n:

            tobj[b, gi, gj] = 1 
            nonempty, count = torch.unique(b, return_counts=True) 
            for k, bi in enumerate(nonempty):
                tnum[bi, :, :] = count[k] 
        
        obji = self.BCEobj(p[:, 0, :, :], tobj)
        numi = self.MSELoss(p[:, 1, :, :], tnum)

        lobj += obji  
        lnum += numi

        bs = tobj.shape[0]  # batch size

        return (lobj * bs) + lnum, lobj.detach(), lnum.detach()


    def build_targets(self, p, targets):
        # Build targets for compute_loss()
        indices = []

        gain = torch.ones(3, device=targets.device)  # normalized to gridspace gain
        
        gain[1:3] = torch.tensor(p.shape)[[3, 2]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  # image, class
        gxy = t[:, 1:]  # grid xy


        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices.append((b, gj, gi))  


        return indices