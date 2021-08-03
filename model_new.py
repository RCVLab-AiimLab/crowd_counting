
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
    def __init__(self, load_weights=False, in_size=128):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up = nn.Upsample((8,8), mode='bilinear')
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

                # a = len(list(self.frontend.state_dict().items())[0][1][:,:3])
                # print(a)
                ## for depth activated training, activate the 4 lines below:
                # if i==0:
                #     list(self.frontend.state_dict().items())[i][1].data[:,:3] = list(mod.state_dict().items())[i][1].data[:,:3]
                # else:
                #     list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

        self.in_size = (64 // 8)**2
        self.regress = nn.Sequential(nn.Linear(self.in_size, self.in_size//2), 
                                    nn.LeakyReLU(),
                                    nn.Linear(self.in_size//2, 1))


    def forward(self, x, training=True):
        x = self.frontend(x)
        # xd = nn.cat(x,d,dim=1)
        x = self.backend(x)
        x = self.output_layer(x)
        x = self.up(x)
        x = x.squeeze(1)
        x = x.view(-1, self.in_size)
        x = self.regress(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  


class Model(nn.Module):
    def __init__(self, model_file, in_size): 
        super(Model, self).__init__()

        with open(model_file) as f:
            self.model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        self.model, self.save = parse_model(deepcopy(self.model_dict), in_size)  

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


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, in_size=128):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device

        # Define criteria
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=device))
        self.in_size = in_size

    def __call__(self, p, targets): 
        device = targets.device
        #lobj = torch.zeros(1, device=device)
        lcell = torch.zeros(1, device=device)
        #lcountInCell = torch.zeros(1, device=device)
        #lcountInNbr = torch.zeros(1, device=device)
        # print('before build', targets.shape[1])
        if targets.shape[1]==3:
            # print(p.shape)
            lcell += self.MSELoss(p, torch.zeros_like(p))
        else:
            indices, _ = self.build_targets(p, targets) 
            # print('indices', indices)
            # print('*'*1000)
            #p0 = p[..., 0]
            #p1 = p[..., 1]
            
            # Losses
            b, gj, gi = indices[0]  
            # print('b', b)
            tobj = torch.zeros_like(p, device=device)  
            tcell = torch.zeros_like(p, device=device)   

            n = b.shape[0]  
            if n:
                #tobj[b, gi, gj] = 1 
                
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
                    #tcell[bi, :, :] = float(count[k])
                    # print('count',count)
                    # print(nonempty)
                    # print(k)
                    # print(bi)
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

            #lobj += self.BCEcls(p0, tobj)
            lcell += self.MSELoss(p, tcell)
            
            bs = tobj.shape[0]
        
        return  lcell, lcell.detach()


    def build_targets(self, p0, targets):
        # print(targets.shape)
        # Build targets for compute_loss()
        indices = []

        gain = torch.ones(5, device=targets.device)  # normalized to gridspace gain
        
        gain[1:3] = torch.tensor((self.in_size//8, self.in_size//8)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
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

