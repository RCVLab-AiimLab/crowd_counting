
import numpy as np 
import math
from copy import deepcopy
import yaml 
import torch
import torch.nn as nn 
import cv2 
from itertools import product, starmap


        
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
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if isinstance(m, Detect):
                x = m(x, not training)  # run
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x


    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


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

    ch, nc, gd, gw = [d['ch']], d['nc'], d['depth_multiple'], d['width_multiple']
    no = 2  # number of outputs
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
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int): 
                args[1] = [list(range(args[1] * 2))] * len(f)
            #args = [args[2]]
        
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2

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
    def __init__(self, nc, ext, ch=()):
        super(Detect, self).__init__() 
        self.no = 2  # number of outputs
        self.m = nn.ModuleList(nn.Conv2d(x, self.no, 1, dilation=1) for x in ch)  # output conv
        self.regress = nn.Sequential(nn.Linear(ch[0], ch[0]//2), 
                                    nn.LeakyReLU(),
                                    nn.Linear(ch[0]//2, 1))
        
        self.ch = ch
        #self.dropout1 = nn.Dropout2d(0.25)

    def forward(self, x, inference):

        x = self.m[0](x[0])  # conv
        bs, _, ny, nx = x.shape  
        x = x.view(bs, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()
        
        x0 = x[..., 0]
        x1 = x[..., 1]
        x1 = x1.view(-1, self.ch[0])
        x1 = self.regress(x1)

        if inference:  
            #x[..., 0] = x[..., 0].sigmoid()
            x0 = x0.sigmoid()

        return x0, x1


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
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.MSELoss = nn.MSELoss()
        

    def __call__(self, p0, p1, targets): 
        device = targets.device
        lobj = torch.zeros(1, device=device)
        lcell = torch.zeros(1, device=device)
        lcountInCell = torch.zeros(1, device=device)
        lcountInNbr = torch.zeros(1, device=device)
        indices, neighbors = self.build_targets(p0, targets) 

        # Losses
        b, gj, gi = indices[0]  
        tobj = torch.zeros_like(p0, device=device)  
        tcell = torch.zeros_like(p1, device=device)   

        n = b.shape[0]  
        if n:
            tobj[b, gi, gj] = 1 

            nonempty, count = torch.unique(b, return_counts=True) 
            '''
            nbrs_coord = []
            for bi in nonempty:
                ind = torch.where(b==bi)[0]
                nbrs_coord.append(neighbors[ind[0], :])
            nbrs_coord = torch.stack(nbrs_coord)
            nbrs_cell = torch.ones((len(nbrs_coord), 8)) * (-1)
            for bi in range(len(nbrs_coord)):
                coords = nbrs_coord[bi].view(-1, 2)
                for i in range(len(nbrs_coord)):
                    for j in range(1,9):
                        if nbrs_coord[i, 0] == coords[j, 0] and nbrs_coord[i, 1] == coords[j, 1]:
                            nbrs_cell[bi, j-1] = nonempty[i]
            
            
            '''
            for k, bi in enumerate(nonempty):
                #indx = torch.where(b==bi)
                #tcell[bi, gi[indx], gj[indx]] = float(count[k])
                tcell[bi, 0] = float(count[k])
            '''
                #tcell[bi, :, :] = float(count[k])
                #countInCell = p[bi, gi[indx], gj[indx], 1].mean()
                #countInCell_gt = tcell[bi, gi[indx], gj[indx]].mean()
                #lcountInCell += torch.abs(countInCell - countInCell_gt).mean()  
                
                nbr = nbrs_cell[k, nbrs_cell[k,:] >= 0]
                CountInNbr_gt = torch.clone(count[k])
                print(p[bi, gi[indx], gj[indx], 1])
                countInNbr = torch.clone(p[bi, gi[indx], gj[indx], 1].mean())
                for i in nbr:
                    ind = torch.where(nonempty==i)[0][0]
                    CountInNbr_gt += torch.clone(count[ind])
                    countInNbr += torch.clone(p[ind, gi[indx], gj[indx], 1].mean())

                lcountInNbr += torch.abs(countInNbr - CountInNbr_gt).mean()
            '''

        lobj += self.BCEobj(p0, tobj)
        lcell += torch.abs(p1 - tcell).mean()
        #lcell += self.MSELoss(p1, tcell)
        #countInCell = p[..., 1].view(p.size(0), -1).mean(1, keepdim=True).sum()
        #countInCell_gt = tcell.view(p.size(0), -1).mean(1, keepdim=True).sum()
        #lcountInCell += torch.abs(countInCell - countInCell_gt).mean()  

        bs = tobj.shape[0]  # batch size

        return (lobj * bs) + (lcell) + lcountInNbr, lobj.detach(), lcell.detach(), lcountInNbr.detach()


    def build_targets(self, p0, targets):
        # Build targets for compute_loss()
        indices = []

        gain = torch.ones(5, device=targets.device)  # normalized to gridspace gain
        
        gain[1:3] = torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  # image, class
        gxy = t[:, 1:3]  # grid xy


        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices.append((b, gj, gi))  

        nij = t[:, 3:]
        bi, bj = nij.T
        cells = starmap(lambda a,b: (bi+a, bj+b), product((0,-1,+1), (0,-1,+1)))
        nbr = list(cells)
        neighbors = []
        '''
        for i in nbr: 
            for j in i: 
                neighbors.append(j)

        neighbors = torch.stack(neighbors).T
        '''

        return indices, neighbors

