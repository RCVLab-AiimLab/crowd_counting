import torch.nn as nn
import torch
from torchvision import models
<<<<<<< Updated upstream
from utils import save_net,load_net
from caps import *
from caps2 import *
import torch.nn.functional as F
=======
# import segmentation_models_pytorch as smp 
>>>>>>> Stashed changes


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
<<<<<<< Updated upstream
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.avg = nn.AvgPool2d(kernel_size=28)
=======

        self.backend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend_feat_depth  = [64, 'M', 128, 'M', 256]
        self.backend = make_layers(self.backend_feat, in_channels=3)
        self.frontend = make_layers(self.frontend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.frontend_depth = make_layers(self.frontend_feat_depth, in_channels=1)

>>>>>>> Stashed changes
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, 512)

        self.comp_feat = [32,1]
        self.frontend = make_layers(self.frontend_feat)
        self.caps = CapsNet(3)
        self.comp_mod = make_layers_comp(self.comp_feat)
        # self.caps_att = make_layers_caps_att()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
<<<<<<< Updated upstream
        self.network = CapsuleNetwork(image_width=28,
                         image_height=28,
                         image_channels=512,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=10, # one for each MNIST digit
                         output_unit_size=output_unit_size).cuda()
=======
        self.up = nn.Upsample((8,8), mode='bilinear')
>>>>>>> Stashed changes
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
<<<<<<< Updated upstream
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        # x_att = self.conv_att(x).unsqueeze(-1).unsqueeze(-1)
        # x_att = self.avg(x)

<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes

        x_att = F.adaptive_avg_pool2d(x, (1, 1))
        x_att = self.flatten(x_att)
        x_att = self.linear(x_att)
        x_att = F.relu(x_att)
        x_att = self.linear(x_att)
        x_att = F.tanh(x_att)
        x_att = x_att.unsqueeze(-1).unsqueeze(-1)
        x_att = x_att * x
        x_att = x_att + x

<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes

        # print(x.shape)
        # x_att = self.caps(x_att)[0]

        x_att = self.network(x_att)
        # print(x_att.shape)
        
        # print(x_att.shape)
        # x_caps_att = x_conv_att*x_att[0]
        # x_conv_att = x_conv_att.view((1,x_conv_att.shape[3],x_conv_att.shape[1],x_conv_att.shape[2]))
        # x_caps_att = x_caps_att.reshape((1,x_caps_att.shape[3],x_caps_att.shape[1],x_caps_att.shape[2]))
        # x_att = torch.cat((x_conv_att,x_caps_att),dim=1)

        x_att = self.comp_mod(x_att)
        return x_att
=======
            for i in range(len(self.backend.state_dict().items())):
                list(self.backend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        
        
        bilinear = False
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.outc1 = OutConv(256, 1)
        self.outc2 = OutConv(128, 1)
        
        self.info(verbose=verbose)


    def forward(self, x, x_depth, training=True):
        device = x.device
        x = self.backend(x)
        # print('start',x_depth[0])
        x_depth  = self.frontend_depth(x_depth)
        # print('after front end',x_depth[0])
        x_depth = nn.AvgPool2d((x_depth.shape[2], x_depth.shape[3]))(x_depth)
        # print(x_depth.shape)
        x_depth  = nn.Flatten()(x_depth)
        # print(x_depth.shape)
        x_depth = nn.Linear(256, 128, device=device)(x_depth)
        # print(x_depth.shape)
        x_depth = nn.Linear(128, 8, device=device)(x_depth)
        x_depth = nn.Linear(8, 3, device=device)(x_depth)
        # print(x_depth)
        x_depth = nn.Softmax()(x_depth)
        # print('x_depth', x_depth.shape)
        # print('x', x.shape)
        x0 = self.frontend(x)
        x0 = self.output_layer(x0) 
        x0 = x0.squeeze(1) # count_0

        x = self.up1(x)
        x1 = self.outc1(x)
        x1 = x1.squeeze(1) # count_1

        x2 = self.up2(x) 
        x2 = self.outc2(x2) 
        x2 = x2.squeeze(1) # count_2
        # print('x_depth 0', x_depth[:, 0].shape)
        # print('x0', x0.shape)
        # print('x 0', x0.sum(dim = [1, 2]).shape)
        count = (x0.sum(dim = [1, 2]) * x_depth[:, 0]) + (x1.sum(dim = [1, 2]) * x_depth[:, 1]) + (x2.sum(dim = [1, 2]) * x_depth[:, 2])
        # print(x_depth)
        # x = torch.stack([x0, x1, x2])
        # print('x', x.shape)
        '''# uncomment if binary loss is used
        if not training: 
            x0 = x0.sigmoid()
            x1 = x1.sigmoid()
            x2 = x2.sigmoid()
        '''    
        return x0, x1, x2, count


>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    return nn.Sequential(*layers)                 
=======
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

<<<<<<< Updated upstream
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
=======
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def __call__(self, p0, p1, p2, targets, count): 
        device = targets.device
        
        lcount = torch.zeros(1, device=device)
        lcount_0 = torch.zeros(1, device=device)
        lcount_1 = torch.zeros(1, device=device)
        lcount_2 = torch.zeros(1, device=device)

        indices_0, indices_1, indices_2 = self.build_targets(targets) 

        tcount_0 = torch.zeros_like(p0, device=device)  
        tcount_1 = torch.zeros_like(p1, device=device)   
        tcount_2 = torch.zeros_like(p2, device=device)  
        
        # Losses
        b, gj, gi = indices_0[0]  
 
        n = b.shape[0]  
        if n:
            
            tcount_0[b, gi, gj] = 1 

            b, gj, gi = indices_1[0]  
            tcount_1[b, gi, gj] = 1 

            b, gj, gi = indices_2[0]  
            tcount_2[b, gi, gj] = 1 

        lcount_0 += self.MSELoss(p0, tcount_0)
        lcount_1 += self.MSELoss(p1, tcount_1)
        lcount_2 += self.MSELoss(p2, tcount_2)
        lcount += self.MSELoss(torch.sum(p0, (-1, -2)) + torch.sum(p1, (-1, -2)) + torch.sum(p2, (-1, -2)), count)
    
        return  (lcount_0 + lcount_1 + lcount_2 + lcount), lcount_0.detach(), lcount_1.detach(), lcount_2.detach()
>>>>>>> Stashed changes


    def build_targets(self, p0, targets):
        # print(targets.shape)
        # Build targets for compute_loss()
        indices = []

        gain = torch.ones(5, device=targets.device)  # normalized to gridspace gain
        
        gain[1:3] = torch.tensor((self.in_size//8, self.in_size//8)) #torch.tensor(p0.shape)[[2, 1]]  # xyxy gain
        t = targets * gain

        b = t[:, 0].long().T  # image, class
        gxy = t[:, 1:3]  # grid xy

>>>>>>> Stashed changes

def make_layers_conv_att(cfg, in_channels = 3, batch_norm=False, dilation = False):
    layers = []
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=28), nn.Flatten()]
        else:
            layers += [nn.Linear(in_features = 512, out_features = 512)]
            in_channels = v
    return nn.Sequential(*layers)  

def make_layers_comp(cfg, in_channels = 64,batch_norm=True,dilation = False):
    layers = []
    for v in cfg:
        print(v)
        print(in_channels)
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv1 = nn.Conv2d(in_channels, v, kernel_size=1)
            # conv2 = nn.Conv2d(32, v, kernel_size=1)
            layers += [conv1, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  


            

            