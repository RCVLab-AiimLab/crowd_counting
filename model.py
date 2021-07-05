#Max poolings in CSRNet are replaced with convolutional layers with stride=2

import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net

#Implementation of Capsule attention module
#The shapes are matched for Crala dataset and CSRNet
#The input shape is (batch,512,120,67) and the output shape is (batch,64,120,67)
#It can be used in the middel of CSRNet frontend and backend

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F


class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()

        self.conv_size = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=9, # fixme constant
                               stride=2,
                               bias=True)
        
        self.up = nn.Upsample(size=(33, 24), mode='bilinear')

        
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=9, # fixme constant
                               stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_size(x)
        x = self.up(x)
        x = self.conv0(x)
        return self.relu(x)

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)

    def forward(self, x):
        return self.conv0(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.FC = nn.Linear(160,450)

        

        self.up = nn.Upsample(scale_factor=2, mode='bicubic')

        self.up2 = nn.Upsample(size=(120,67), mode='bicubic')


        self.conv1 = nn.Conv2d(in_channels= 1,
                               out_channels=16,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=1, # fixme constant
                               padding ='same',
                               bias=True)
        
        self.conv2 = nn.Conv2d(in_channels= 16,
                               out_channels=64,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=1, # fixme constant
                               padding ='same',
                               bias=True)
                
                

    def forward(self, x):
        x = self.FC(x)
        x = torch.reshape(x, (-1,1,30,15))
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)
        x = self.up(x)
        x = self.up2(x)

        return x


class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.num_units)]

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1)

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)

        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).cuda()

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)




class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CapsuleNetwork, self).__init__()

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height


        self.l1 = CapsuleConvLayer(in_channels=conv_inputs,
                                    out_channels=conv_outputs)

        self.l2 = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.l3 = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)
        
        self.l4 = nn.Flatten(start_dim=1)
        self.l5 = nn.ReLU()
        self.l6 = nn.Tanh()
        self.l7 = Decoder()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x
        



conv_inputs = 512
conv_outputs = 256
num_primary_units = 16
primary_unit_size = 32 * 6 * 6  # fixme get from conv2d
output_unit_size = 16

Capsule_attention_module = CapsuleNetwork(image_width=28,
                         image_height=28,
                         image_channels=512,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=10, # one for each MNIST digit
                         output_unit_size=output_unit_size).cuda()
print(Capsule_attention_module)


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, [64,'M'], 128, [128, 'M'], 256, 256, [256, 'M'], 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x1 = self.frontend(x)
        x2 = Capsule_attention_module(x1)
        x = torch.cat((x1,x2),1)
        x = self.backend(x)
        x = self.output_layer(x)
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
        if isinstance(v, list):
            layers += [nn.Conv2d(in_channels, v[0], kernel_size=3, padding=d_rate,dilation = d_rate, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True), nn.Dropout2d(p=0.2)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True), nn.Dropout2d(p=0.2)]
            in_channels = v
    return nn.Sequential(*layers) 
